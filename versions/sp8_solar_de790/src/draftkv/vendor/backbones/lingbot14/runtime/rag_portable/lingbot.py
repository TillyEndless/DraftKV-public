from contextlib import contextmanager
from contextvars import ContextVar
import json
import os
from pathlib import Path

import torch

from .base import host_copy
from .free_cache import PortableFreeKVCache


_CLEAN = ContextVar('lingbot_clean_commit', default=False)
_CACHES = []


@contextmanager
def attention_phase(clean):
    token = _CLEAN.set(clean)
    try:
        yield
    finally:
        _CLEAN.reset(token)


def _span(bank, start, stop, device):
    if start == stop:
        return bank['offloaded'][0][:, :, :0].to(device)
    chunk_tokens = bank['offloaded'][0].shape[-2]
    parts = []
    for chunk in range(start // chunk_tokens, (stop - 1) // chunk_tokens + 1):
        lower = max(start - chunk * chunk_tokens, 0)
        upper = min(stop - chunk * chunk_tokens, chunk_tokens)
        parts.append(bank['offloaded'][chunk][:, :, lower:upper])
    return torch.cat(parts, dim=-2).to(device)


@torch.no_grad()
def route_attention(module, query, key, value, native_keys, native_values, current_start, frame_tokens,
                    raw_key=None, grid_sizes=None, freqs=None, sp_size=1, raw_query=None):
    if os.environ.get('LINGBOT_RAG', 'off') not in ('off', 'freekv', 'shadowkv', 'worldkv', 'infllm', 'draftmap'):
        raise ValueError('Unsupported LingBot RAG method; refusing a silently native result')
    if os.environ.get('LINGBOT_RAG') == 'draftmap':
        from .draftmap_route import route_draftmap
        return route_draftmap(module, query, key, value, native_keys, native_values, current_start,
                              frame_tokens, raw_key, raw_query, sp_size, _CLEAN.get(), grid_sizes=grid_sizes, freqs=freqs)
    if os.environ.get('LINGBOT_RAG') == 'infllm':
        from .infllm_route import route_infllm
        return route_infllm(module, query, key, value, native_keys, native_values, current_start,
                            frame_tokens, raw_key, raw_query, grid_sizes, freqs, sp_size, _CLEAN.get())
    if os.environ.get('LINGBOT_RAG') == 'worldkv':
        from .world_route import route_world
        return route_world(module, query, key, value, native_keys, native_values, current_start,
                           frame_tokens, sp_size, _CLEAN.get())
    if os.environ.get('LINGBOT_RAG') == 'shadowkv':
        from .shadow_route import route_shadow
        return route_shadow(module, query, key, value, native_keys, native_values, current_start,
                            frame_tokens, raw_key, grid_sizes, freqs, sp_size, _CLEAN.get())
    if os.environ.get('LINGBOT_RAG') != 'freekv':
        return native_keys, native_values
    if query.shape != key.shape or key.shape != value.shape or query.shape[0] != 1:
        raise ValueError('Expected matching batch-one head-sharded Q/K/V')
    chunk_tokens = key.shape[1]
    if current_start % chunk_tokens or chunk_tokens % frame_tokens:
        raise ValueError('Chunk offsets must use the unpadded native token grid')
    cache = getattr(module, '_portable_freekv', None)
    if cache is None:
        cache = PortableFreeKVCache()
        cache.banks[0] = dict(minimum=[], maximum=[], offloaded=[], resident_ids=None,
                             pending=None, last_query=None)
        module._portable_freekv = cache
        _CACHES.append(cache)
    bank = cache.banks[0]
    if len(bank['offloaded']) * chunk_tokens != current_start:
        raise RuntimeError('Noisy KV was committed, or a clean commit is missing')
    if hasattr(cache, 'active_call'):
        raise RuntimeError('Previous attention was not finalized')
    total_tokens = module.local_attn_size * frame_tokens
    sink_tokens = module.sink_size * frame_tokens
    local_tokens = chunk_tokens
    retrieval_budget = total_tokens - chunk_tokens - sink_tokens - local_tokens
    if retrieval_budget <= 0 or retrieval_budget % cache.block_size:
        raise ValueError('Native window cannot fit sink/current/local plus complete retrieval pages')
    cache.stats.update(history_budget_frames=module.local_attn_size - chunk_tokens // frame_tokens,
                       sink_frames=module.sink_size, local_frames=chunk_tokens // frame_tokens,
                       head_shard=query.shape[2], rows_per_frame=frame_tokens)
    current_query = query.transpose(1, 2)
    retrieved = current_start + chunk_tokens > total_tokens
    cache.active_call = (key.detach(), value.detach(), current_query, retrieval_budget, _CLEAN.get(), retrieved)
    if not retrieved:
        return native_keys, native_values
    if native_keys.shape[1] != total_tokens:
        raise ValueError('Native max_attention_size and declared rolling window differ')
    cache.promote(sink_tokens, current_start - local_tokens, query.device)
    ids, global_keys, global_values = cache.select_fetch(0, current_query, retrieval_budget)
    if bool((ids < sink_tokens).any()) or bool((ids >= current_start - local_tokens).any()):
        raise RuntimeError('Retrieved sink, local or future token')
    sink = _span(bank, 0, sink_tokens, query.device)
    local = _span(bank, current_start - local_tokens, current_start, query.device)
    keys = torch.cat((sink[0][None], global_keys, local[0][None], key.transpose(1, 2)), dim=-2)
    values = torch.cat((sink[1][None], global_values, local[1][None], value.transpose(1, 2)), dim=-2)
    if keys.shape[-2] != total_tokens:
        raise RuntimeError('Native total attention budget changed')
    cache.stats['retrieved_forwards'] += 1
    cache.stats['forward_history'].append(dict(current_start=current_start, clean=_CLEAN.get(),
                                               sink_tokens=sink_tokens, local_tokens=local_tokens,
                                               retrieved_tokens=ids.shape[-1], current_tokens=chunk_tokens))
    return keys.transpose(1, 2).contiguous(), values.transpose(1, 2).contiguous()


def finish_attention(module):
    if os.environ.get('LINGBOT_RAG') == 'draftmap':
        from .draftmap_route import finish_draftmap
        return finish_draftmap(module)
    if os.environ.get('LINGBOT_RAG') == 'infllm':
        from .infllm_route import finish_infllm
        return finish_infllm(module)
    if os.environ.get('LINGBOT_RAG') == 'worldkv':
        from .world_route import finish_world
        return finish_world(module)
    if os.environ.get('LINGBOT_RAG') == 'shadowkv':
        from .shadow_route import finish_shadow
        return finish_shadow(module)
    if os.environ.get('LINGBOT_RAG') != 'freekv':
        return
    cache = module._portable_freekv
    key, value, query, budget, clean, retrieved = cache.active_call
    del cache.active_call
    if retrieved:
        cache.after_attention(0, query, budget, clean)
    if clean:
        cache.banks[0]['offloaded'].append(host_copy(torch.stack((key[0], value[0])).transpose(1, 2).contiguous()))
        cache.stats['clean_commits'] += 1


def dump_stats(directory):
    output = Path(directory)
    output.mkdir(parents=True, exist_ok=True)
    path = output / f'rank{os.environ.get("RANK", "0")}.json'
    if path.exists():
        raise FileExistsError(path)
    from .shadow_route import SHADOW_CACHES
    from .world_route import WORLD_CACHES
    from .infllm_route import INFLLM_CACHES
    from .draftmap_route import DRAFT_CACHES
    caches = _CACHES + SHADOW_CACHES + WORLD_CACHES + INFLLM_CACHES + DRAFT_CACHES
    path.write_text(json.dumps([cache.stats for cache in caches], indent=2) + '\n')
