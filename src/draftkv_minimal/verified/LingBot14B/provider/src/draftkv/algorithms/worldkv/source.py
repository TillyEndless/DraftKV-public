"""Pinned original WorldKV source and private portable-provider instances."""
import ast
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import types
from uuid import uuid4

from draftkv.algorithms.shadowkv.source import REPO, settings, verify_sources
from draftkv.backbones.contracts import BackboneError

AUTHOR = Path(__file__).with_name('vendor') / 'author'
PORTABLE = REPO / 'src/draftkv/vendor/backbones/lingbot14/runtime/rag_portable'


def preflight(config, root=None):
    checked = verify_sources('worldkv', root)
    if config.get('benchmark') != 'bench2' or config.get('backbone') not in ('LingBot1.3B', 'LingBot14B', 'SolarWM33B'):
        raise BackboneError('BLOCKED_INPUT_PROTOCOL', 'WorldKV requires Bench2 LingBot camera-conditioned native input')
    cfg = settings(config)
    if not 0 < float(cfg.get('keep_ratio', .5)) <= 1:
        raise BackboneError('BLOCKED_CONFIGURATION', 'WorldKV keep_ratio must be in (0,1]')
    expected_policy = ('h3_native_mmrope_camera_prope' if config.get('backbone') == 'SolarWM33B' else 'native_absolute_rope')
    if cfg.get('position_policy', 'native_absolute_rope') != expected_policy:
        raise BackboneError('BLOCKED_ALGORITHM_SEMANTICS', 'frozen LingBot WorldKV preserves native absolute Wan RoPE; SolarWM/H3 PRoPE cannot be substituted')
    # Both native LingBot models use W18/S6/C4. The original compression takes
    # chunk_size explicitly: no fixed C3 reshape or grouping is introduced.
    return dict(status='READY', algorithm='worldkv', source_files=checked,
                consumer='native_fa2', native_chunk_frames=5 if config.get('backbone') == 'SolarWM33B' else 4,
                camera_policy='author_absolute_interpolated_pose_distance',
                position_policy=expected_policy,
                prope_scope='SolarWM_H3_only_not_LingBot_checkpoint', video_e2e=False)


def load_provider():
    """One module namespace per owner; frozen _POSES/cache globals never cross CFG."""
    prefix = '_draftkv_worldkv_' + uuid4().hex
    package = types.ModuleType(prefix)
    package.__path__ = [str(PORTABLE)]
    sys.modules[prefix] = package
    for name in ('base', 'world_core', 'world_route'):
        spec = importlib.util.spec_from_file_location(prefix + '.' + name, PORTABLE / (name + '.py'))
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    return sys.modules[prefix + '.world_route']


def mapped_compression(record):
    """Add an index observation to the AUTHOR AST, preserving every math node.

    The callback observes the original pooled topk indices before gather. It
    never computes another ranking, changes ties, or infers IDs from KV values.
    """
    import torch
    path = AUTHOR / 'wan/modules/model_fast.py'
    tree = ast.parse(path.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_compress_retrieval_kv')

    class Observe(ast.NodeTransformer):
        def visit_Assign(self, node):
            if any(isinstance(t, ast.Name) and t.id == 'gather_idx' for t in node.targets):
                callback = ast.Expr(ast.Call(ast.Name('_record', ast.Load()),
                    [ast.Name(n, ast.Load()) for n in ('ci', 'idx', 'anchor_off', 'chunk_size', 'frame_seqlen', 'pooled')], []))
                return [callback, node]
            return node
    function = Observe().visit(function)
    namespace = dict(torch=torch, math=math, _record=record)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])), str(path), 'exec'), namespace)
    return namespace[function.name]


def capture_native_camera(pipeline, callback):
    """Observe native interpolation before its relative/Plucker conversion.

    Only an extra callback statement is inserted into the instance's generate;
    the source, interpolation recipe and all original expressions stay intact.
    """
    import inspect
    import textwrap
    method = '_generate_causal_fast' if getattr(pipeline, 'infer_mode', 'causal_fast') == 'causal_fast' else '_generate_causal_pretrain'
    if not hasattr(pipeline, method):
        method = 'generate'  # Older audited snapshots expose one method directly.
    original = getattr(pipeline, method).__func__
    tree = ast.parse(textwrap.dedent(inspect.getsource(original)))
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef))
    function.decorator_list = []
    hits = 0

    class Capture(ast.NodeTransformer):
        def visit_Assign(self, node):
            nonlocal hits
            if (any(isinstance(t, ast.Name) and t.id == 'c2ws_infer' for t in node.targets)
                    and isinstance(node.value, ast.Call)
                    and isinstance(node.value.func, ast.Name)
                    and node.value.func.id == 'interpolate_camera_poses'):
                hits += 1
                return [node, ast.Expr(ast.Call(ast.Name('_worldkv_camera', ast.Load()),
                                              [ast.Name('c2ws_infer', ast.Load())], []))]
            return node
    Capture().visit(function)
    if hits != 1:
        raise BackboneError('SOURCE_ABI_DRIFT', 'native generate must expose exactly one absolute camera interpolation seam')
    namespace = dict(original.__globals__, _worldkv_camera=callback)
    exec(compile(ast.fix_missing_locations(tree), inspect.getsourcefile(original) or '<native-camera>', 'exec'), namespace)
    setattr(pipeline, method, types.MethodType(namespace[function.name], pipeline))
    return hashlib.sha256(inspect.getsource(original).encode()).hexdigest()
