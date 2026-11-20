"""Source-guarded Wan VAE output storage; decoder mathematics stay native.

The locked decoder concatenates its growing CUDA output after every latent
frame. Keep those exact output bytes in pageable CPU storage instead, then
restore the original device/dtype after the native feature-cache cleanup.
Only an individual WanVAE_ instance is adapted; no vendor file or torch global
is changed. Importing this module does not import torch or initialize CUDA.
"""
from __future__ import annotations

import ast
import functools
import hashlib
import inspect
from pathlib import Path
import textwrap
import types

from .contracts import BackboneError


# Entire locked WanVAE_.decode AST, including scale, decoder calls and cleanup.
# Whitespace/comments are immaterial; any executable source drift fails closed.
_DECODE_AST_SHA256 = "4b6bcb43e8f657585218190f71ef1085e3a5891176cdb4c536ada5a9b62c368c"
_MARKER = "_draftkv_vae_cpu_staging"


def _drift(detail):
    raise BackboneError("SOURCE_ABI_DRIFT", "Wan VAE CPU staging: " + detail)


class _CPUOutput:
    """One decode's exact 1 + 4*(latent_steps-1) output, without GPU growth."""

    def __init__(self, latent_steps, trace=None):
        if type(latent_steps) is not int or latent_steps < 1:
            _drift("positive native latent-step count required")
        self.latent_steps = latent_steps
        self.frames = 1 + 4 * (latent_steps - 1)
        self.steps = self.offset = 0
        self.output = None
        self.device = self.dtype = self.geometry = None
        self.trace = trace

    def append(self, chunk):
        import torch
        if (not isinstance(chunk, torch.Tensor) or chunk.ndim != 5
                or chunk.layout != torch.strided or not chunk.is_floating_point()
                or chunk.requires_grad or self.steps >= self.latent_steps):
            _drift("expected detached native B,C,T,H,W output")
        expected_frames = 1 if self.steps == 0 else 4
        geometry = (chunk.shape[0], chunk.shape[1], chunk.shape[3], chunk.shape[4])
        if (chunk.shape[2] != expected_frames or any(n < 1 for n in geometry)
                or chunk.shape[1] != 3):
            _drift("native decoder must emit one first frame and four later frames")
        if self.output is None:
            self.device, self.dtype, self.geometry = chunk.device, chunk.dtype, geometry
            b, c, h, w = geometry
            self.output = torch.empty((b, c, self.frames, h, w), dtype=self.dtype,
                                      device="cpu", pin_memory=False)
            if self.trace:
                self.trace("vae_cpu_staging_begin", latent_steps=self.latent_steps,
                           decoded_frames=self.frames, dtype=str(self.dtype),
                           output_device=str(self.device), storage="pageable_cpu")
        elif (chunk.device != self.device or chunk.dtype != self.dtype
              or geometry != self.geometry):
            _drift("native output device, dtype or spatial geometry changed")
        # Matching dtype, blocking D2H: no cast, normalization or arithmetic.
        self.output.narrow(2, self.offset, expected_frames).copy_(chunk, non_blocking=False)
        self.offset += expected_frames
        self.steps += 1

    def finish(self):
        if (self.output is None or self.steps != self.latent_steps
                or self.offset != self.frames):
            _drift("native output accumulation is incomplete")
        # Called only after the original self.clear_cache(). The outer native
        # Wan2_1_VAE.decode still owns float(), clamp_(), squeeze() and encoding.
        result = self.output.to(device=self.device, dtype=self.dtype, non_blocking=False)
        if self.trace:
            self.trace("vae_cpu_staging_complete", latent_steps=self.steps,
                       decoded_frames=self.offset, dtype=str(result.dtype),
                       output_device=str(result.device),
                       staged_bytes=self.output.numel() * self.output.element_size())
        return result


def _code_identity(code):
    """Ignore filenames/line tables, but reject an in-memory replaced function."""
    constants = tuple(_code_identity(value) if isinstance(value, types.CodeType) else value
                      for value in code.co_consts)
    return (code.co_code, constants, code.co_names, code.co_varnames,
            code.co_argcount, code.co_posonlyargcount, code.co_kwonlyargcount,
            code.co_freevars, code.co_cellvars, code.co_flags)


def _compile_decode(original, *, trace=None, source_text=None):
    try:
        source = textwrap.dedent(source_text if source_text is not None else inspect.getsource(original))
        tree = ast.parse(source)
    except (OSError, TypeError, SyntaxError) as exc:
        raise BackboneError("SOURCE_ABI_DRIFT", "Wan VAE CPU staging: native decode source unavailable") from exc
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        _drift("expected one native decode function")
    function = tree.body[0]
    # Python 3.12 adds the empty type_params field to an ordinary function.
    # Normalize that non-executable addition, but reject actual type parameters.
    # Python 3.13 omits empty fields by default; retain the same full AST
    # representation on every supported interpreter without relaxing the pin.
    options = {"show_empty": True} if "show_empty" in inspect.signature(ast.dump).parameters else {}
    canonical = ast.dump(function, include_attributes=False, **options).replace(", type_params=[]", "")
    digest = hashlib.sha256(canonical.encode()).hexdigest()
    if digest != _DECODE_AST_SHA256:
        _drift("locked decode source/signature changed")
    filename = inspect.getsourcefile(original) or "<native-wan-vae>"
    namespace = dict(original.__globals__)
    # Source inspection alone could mistake an in-memory patched callable for
    # its unchanged file. Compare its executable code against that source too.
    exec(compile(tree, filename, "exec", dont_inherit=True), namespace)
    if _code_identity(namespace[function.name].__code__) != _code_identity(original.__code__):
        # CPython 3.12 emits different NULL-call placement when a module import
        # is known to the compiler. Validate against the complete locked native
        # module, without executing it; do not weaken executable identity checks.
        try:
            module_code = compile(Path(filename).read_text(), filename, "exec", dont_inherit=True)
        except (OSError, SyntaxError) as exc:
            raise BackboneError("SOURCE_ABI_DRIFT",
                                "Wan VAE CPU staging: native module source unavailable") from exc
        def code_objects(code):
            yield code
            for value in code.co_consts:
                if isinstance(value, types.CodeType):
                    yield from code_objects(value)
        candidates = [code for code in code_objects(module_code)
                      if getattr(code, "co_qualname", None) == original.__qualname__]
        if (len(candidates) != 1
                or _code_identity(candidates[0]) != _code_identity(original.__code__)):
            _drift("native decode callable differs from its locked source")
    loop = next(node for node in function.body if isinstance(node, ast.For))
    branch = loop.body[1]
    # Full-function hash above fixes these exact nodes; also count the growing
    # output cat, so this can never rewrite a decoder's causal feature-cache cat.
    cats = [node for node in ast.walk(function)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name) and node.func.value.id == "torch"
            and node.func.attr == "cat"]
    if len(cats) != 1 or not isinstance(branch, ast.If):
        _drift("native output accumulation seam changed")
    initialize = ast.parse("_cpu_output = _new_cpu_output(iter_)").body[0]
    function.body.insert(function.body.index(loop), initialize)
    branch.body.extend(ast.parse("_cpu_output.append(out)\ndel out").body)
    branch.orelse[1:] = ast.parse("_cpu_output.append(out_)\ndel out_").body
    returned = function.body[-1]
    if not isinstance(returned, ast.Return) or not isinstance(returned.value, ast.Name) or returned.value.id != "out":
        _drift("native decode return seam changed")
    returned.value = ast.parse("_cpu_output.finish()", mode="eval").body
    namespace["_new_cpu_output"] = lambda steps: _CPUOutput(steps, trace)
    exec(compile(ast.fix_missing_locations(tree), filename, "exec", dont_inherit=True), namespace)
    adapted = functools.wraps(original)(namespace[function.name])
    return adapted, hashlib.sha256(source.encode()).hexdigest()


def install_vae_cpu_staging(vae_model, *, trace=None, source_path=None):
    """Adapt one locked WanVAE_ instance; return False on an intact repeat call.

    ``vae_model`` is ``pipeline.vae.model``, not the Wan2_1_VAE outer wrapper.
    An optional source_path must equal the actual native module's source file.
    """
    current = getattr(vae_model, "decode", None)
    marker = getattr(vae_model, _MARKER, None)
    if marker is not None:
        if (not isinstance(marker, dict) or not isinstance(current, types.MethodType)
                or current.__self__ is not vae_model or current.__func__ is not marker.get("adapted")
                or getattr(type(vae_model), "decode", None) is not marker.get("original")):
            _drift("installed instance binding changed")
        if source_path is not None and Path(source_path).resolve() != marker["source_path"]:
            _drift("native module path changed")
        return False
    if (type(vae_model).__name__ != "WanVAE_" or not isinstance(current, types.MethodType)
            or current.__self__ is not vae_model
            or getattr(type(vae_model), "decode", None) is not current.__func__):
        _drift("expected the unmodified WanVAE_ instance decode method")
    original = current.__func__
    native_path = inspect.getsourcefile(original)
    if not native_path:
        _drift("native decode source is unavailable")
    native_path = Path(native_path).resolve()
    if tuple(native_path.parts[-3:]) != ("wan", "modules", "vae2_1.py"):
        _drift("decode is not from the locked Wan VAE module")
    if source_path is not None and Path(source_path).resolve() != native_path:
        _drift("decode source is outside the planned native module")
    adapted, source_sha = _compile_decode(original, trace=trace)
    vae_model.decode = types.MethodType(adapted, vae_model)
    setattr(vae_model, _MARKER, dict(original=original, adapted=adapted, source_path=native_path))
    if trace:
        trace("lingbot14_vae_output_storage", policy="same_dtype_cpu_staging_restore_device",
              source_sha256=source_sha, source_ast_sha256=_DECODE_AST_SHA256,
              source_path=str(native_path), decoder_math="native_unchanged",
              feature_cache="native_unchanged", outer_decode="native_unchanged")
    return True


def install_lingbot14_vae_cpu_staging(core):
    """Backbone resource policy shared by every LingBot14B algorithm."""
    if core.plan.get("backbone") != "LingBot14B":
        return False
    path = Path(core.plan["native_root"]) / "wan/modules/vae2_1.py"
    return install_vae_cpu_staging(core.pipeline.vae.model, trace=core.trace, source_path=path)


def install_lingbot_vae_cpu_staging(core):
    """Both LingBot checkpoints use the exact same locked native VAE decoder."""
    if core.plan.get("backbone") != "LingBot1.3B":
        return install_lingbot14_vae_cpu_staging(core)
    path = Path(core.plan["native_root"]) / "wan/modules/vae2_1.py"
    def trace(event, **fields):
        if event == "lingbot14_vae_output_storage":
            event = "lingbot_vae_output_storage"
            fields["backbone"] = "LingBot1.3B"
        core.trace(event, **fields)
    return install_vae_cpu_staging(core.pipeline.vae.model, trace=trace, source_path=path)
