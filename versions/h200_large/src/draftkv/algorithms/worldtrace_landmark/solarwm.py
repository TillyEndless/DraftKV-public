"""Keep H3's native prefix attention and replace only its video/cache loop.

WT owns all video history. Running H3's discarded rolling video attention first
would duplicate its cache and rotary/PRoPE intermediates. Prefix queries and the
video output camera basis remain native; the replacement consumes native FA2.
"""
import ast
import inspect
import textwrap

from draftkv.backbones.contracts import BackboneError


def native_prefix_attention(function):
    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    definition = tree.body[0]
    loops = [node for node in definition.body if isinstance(node, ast.For)
             and ast.unparse(node.target) == '(noisy, index)'
             and isinstance(node.iter, ast.Name) and node.iter.id == 'chunk_specs']
    if len(loops) != 1 or not any(
            isinstance(n, ast.Call) and ast.unparse(n.func) == 'control.cache.commit'
            for n in ast.walk(loops[0])):
        raise BackboneError('SOURCE_ABI_DRIFT', 'H3 native video/cache loop changed')
    definition.args.kwonlyargs.append(ast.arg(arg='_worldtrace_video'))
    definition.args.kw_defaults.append(None)
    definition.body[definition.body.index(loops[0])] = ast.parse(
        'outputs.append(_worldtrace_video())').body[0]
    namespace = dict(function.__globals__)
    exec(compile(ast.fix_missing_locations(tree), inspect.getsourcefile(function)
                 or '<native-h3-prefix>', 'exec'), namespace)
    return namespace[definition.name]
