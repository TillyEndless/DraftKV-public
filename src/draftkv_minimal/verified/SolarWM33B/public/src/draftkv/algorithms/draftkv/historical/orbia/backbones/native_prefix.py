# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations
from draftkv.streaming.external import external_symbol, external_path
import ast, inspect, textwrap
from .contracts import BackboneError

def native_prefix_attention(function):
    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    definition = tree.body[0]
    loops = [node for node in definition.body if isinstance(node, ast.For) and ast.unparse(node.target) == '(noisy, index)' and isinstance(node.iter, ast.Name) and (node.iter.id == 'chunk_specs')]
    if len(loops) != 1 or not any((isinstance(n, ast.Call) and ast.unparse(n.func) == 'control.cache.commit' for n in ast.walk(loops[0]))):
        raise BackboneError('SOURCE_ABI_DRIFT', 'H3 native video/cache loop changed')
    definition.args.kwonlyargs.append(ast.arg(arg='_draftkv_video'))
    definition.args.kw_defaults.append(None)
    definition.body[definition.body.index(loops[0])] = ast.parse('outputs.append(_draftkv_video())').body[0]
    namespace = dict(function.__globals__)
    exec(compile(ast.fix_missing_locations(tree), inspect.getsourcefile(function) or '<native-h3-prefix>', 'exec'), namespace)
    return namespace[definition.name]
