# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations
from draftkv.streaming.external import external_symbol, external_path
import ast, hashlib, types
from draftkv.algorithms.draftkv.historical.orbia.backbones.contracts import BackboneError

def capture_native_camera(pipeline, callback):
    """Observe native interpolation before its relative/Plucker conversion.

    Only an extra callback statement is inserted into the instance's generate;
    the source, interpolation recipe and all original expressions stay intact.
    """
    import inspect
    import textwrap
    method = '_generate_causal_fast' if getattr(pipeline, 'infer_mode', 'causal_fast') == 'causal_fast' else '_generate_causal_pretrain'
    if not hasattr(pipeline, method):
        method = 'generate'
    original = getattr(pipeline, method).__func__
    tree = ast.parse(textwrap.dedent(inspect.getsource(original)))
    function = next((n for n in tree.body if isinstance(n, ast.FunctionDef)))
    function.decorator_list = []
    hits = 0

    class Capture(ast.NodeTransformer):

        def visit_Assign(self, node):
            nonlocal hits
            if any((isinstance(t, ast.Name) and t.id == 'c2ws_infer' for t in node.targets)) and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and (node.value.func.id == 'interpolate_camera_poses'):
                hits += 1
                return [node, ast.Expr(ast.Call(ast.Name('_worldkv_camera', ast.Load()), [ast.Name('c2ws_infer', ast.Load())], []))]
            return node
    Capture().visit(function)
    if hits != 1:
        raise BackboneError('SOURCE_ABI_DRIFT', 'native generate must expose exactly one absolute camera interpolation seam')
    namespace = dict(original.__globals__, _worldkv_camera=callback)
    exec(compile(ast.fix_missing_locations(tree), inspect.getsourcefile(original) or '<native-camera>', 'exec'), namespace)
    setattr(pipeline, method, types.MethodType(namespace[function.name], pipeline))
    return hashlib.sha256(inspect.getsource(original).encode()).hexdigest()
