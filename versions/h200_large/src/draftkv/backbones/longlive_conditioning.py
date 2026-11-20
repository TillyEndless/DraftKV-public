"""Select pre-encoded T5 text at the native Wan wrapper boundary, never its loop.

Only cross-attention is invalidated. Native self-KV, clean updates, RNG,
scheduler, local window and VAE are owned by the unchanged author pipeline.
"""
from bisect import bisect_right
from contextvars import ContextVar
import inspect
from types import MethodType

from .contracts import BackboneError
from draftkv.benchmarks.worldattention import native_schedule

# The original DraftKV switch guard stays fail-closed outside this exact
# wrapper transaction. No temporal/sink recache is needed for text-only changes.
_active_text = ContextVar("longlive_native_text_transaction", default=None)


class PromptSwitchGuard:
    def __init__(self, switches):
        self.switches = tuple(switches)

    def __contains__(self, frame):
        if frame not in self.switches:
            return False
        proof = _active_text.get()
        return not (proof and proof[0] == frame and proof[1] == self.switches)


def stage_prompt_embeddings(encoder, prompts, *, device, dtype):
    """One T5 call in one staging pass; preserve every native embedding value.

    Locked WanTextEncoder emits [batch,seq,hidden]; list-emitting native
    encoders are also accepted. WanDiffusionWrapper receives one sample/list.
    """
    encoded = encoder(text_prompts=prompts)
    if set(encoded) != {"prompt_embeds"}:
        raise BackboneError("SOURCE_ABI_DRIFT", "expected native T5 prompt_embeds only")
    values = encoded["prompt_embeds"]
    if len(values) != len(prompts):
        raise BackboneError("SOURCE_ABI_DRIFT", "T5 batch differs from frozen prompt count")
    result = []
    for value in values:
        if getattr(value, "ndim", None) != 2:
            raise BackboneError("SOURCE_ABI_DRIFT", "T5 sample must be [seq,hidden]")
        result.append(value.to(device=device, dtype=dtype))
    return result


class NativePromptSchedule:
    def __init__(self, embeddings, frame_tokens, trace):
        if len(embeddings) != 6 or type(frame_tokens) is not int or frame_tokens <= 0:
            raise BackboneError("BLOCKED_INPUT_PROTOCOL", "six T5 samples and positive native token stride required")
        self.embeddings, self.frame_tokens, self.trace = embeddings, frame_tokens, trace
        self.switches = tuple(native_schedule()["prompt_switch_frames"])
        self.index, self.frame, self.calls = -1, -3, 0
        self.cache = None

    def invoke(self, forward, args, kwargs):
        bound = inspect.signature(forward).bind(*args, **kwargs)
        params = bound.arguments
        start = params.get("current_start")
        if type(start) is not int or start < 0 or start % (3 * self.frame_tokens):
            raise BackboneError("SOURCE_ABI_DRIFT", "native current_start must be C3-aligned token units")
        frame = start // self.frame_tokens
        if frame >= 240 or frame not in (self.frame, self.frame + 3):
            raise BackboneError("SOURCE_ABI_DRIFT", "native chunk skipped or moved backwards")
        cache = params.get("crossattn_cache")
        if not isinstance(cache, list) or not cache or any("is_init" not in c for c in cache):
            raise BackboneError("SOURCE_ABI_DRIFT", "native per-layer crossattention flags required")
        if self.cache is not None and cache is not self.cache:
            raise BackboneError("SOURCE_ABI_DRIFT", "crossattention cache changed during native inference")
        self.cache = cache
        index = bisect_right(self.switches, frame)
        if frame != self.frame:
            if self.frame >= 0 and self.calls != 5:
                raise BackboneError("SOURCE_ABI_DRIFT", "expected four native denoising calls plus one clean call")
            self.frame, self.calls = frame, 0
            if index != self.index:
                if index != self.index + 1:
                    raise BackboneError("BLOCKED_INPUT_PROTOCOL", "native schedule dropped a prompt")
                for layer in cache:
                    layer["is_init"] = False
                self.index = index
                self.trace("native_prompt_switch", prompt_index=index, latent_start=frame,
                           current_start_tokens=start, crossattn_layers_reset=len(cache),
                           self_kv_action="preserved", timing=native_schedule())
        if self.calls >= 5:
            raise BackboneError("SOURCE_ABI_DRIFT", "extra generator call within native chunk")
        clean = bool((params["timestep"] == 0).all())
        if clean != (self.calls == 4):
            raise BackboneError("SOURCE_ABI_DRIFT", "native denoising/clean order changed")
        params["conditional_dict"] = {"prompt_embeds": [self.embeddings[index]]}
        # Set only after the new text is selected and all flags were reset.
        # Each of the five calls keeps that same selected text/proof.
        token = _active_text.set((frame, self.switches))
        try:
            self.trace("native_prompt_call", prompt_index=index, latent_start=frame,
                       call_in_chunk=self.calls, clean=clean)
            result = forward(*bound.args, **bound.kwargs)
        finally:
            _active_text.reset(token)
        self.calls += 1
        return result

    def finish(self):
        if (self.frame, self.index, self.calls) != (237, 5, 5):
            raise BackboneError("BLOCKED_INPUT_PROTOCOL", "native inference did not consume all six prompts/80 chunks")


def bind_native_prompt_schedule(wrapper, embeddings, frame_tokens, trace):
    schedule = NativePromptSchedule(embeddings, frame_tokens, trace)
    original = wrapper.forward

    def forward(owner, *args, **kwargs):
        return schedule.invoke(original, args, kwargs)

    wrapper.forward = MethodType(forward, wrapper)
    return schedule
