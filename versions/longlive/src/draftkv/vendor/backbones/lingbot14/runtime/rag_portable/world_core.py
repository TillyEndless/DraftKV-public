import ast
import copy
from functools import lru_cache
import math
from pathlib import Path

import torch


@lru_cache(maxsize=4)
def compression_function(reference_root):
    path = Path(reference_root) / 'wan/modules/model_fast.py'
    tree = ast.parse(path.read_text())
    method = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                  and node.name == '_compress_retrieval_kv')
    namespace = dict(torch=torch, math=math)
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[method.name]


@lru_cache(maxsize=4)
def camera_selection_code(reference_root):
    path = Path(reference_root) / 'wan/image2video_fast.py'
    tree = ast.parse(path.read_text())

    def assigns(node, name):
        return isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == name
                                                    for target in node.targets)

    parent = next(node for node in ast.walk(tree) if isinstance(node, ast.If)
                  and any(assigns(statement, 'model_sink_size') for statement in node.body))
    start = next(index for index, statement in enumerate(parent.body) if assigns(statement, 'cs'))
    selection = next(node for node in parent.body[start:] if isinstance(node, ast.If)
                     and any(assigns(statement, 'selected_ids') for statement in node.body))
    final = next(index for index, statement in enumerate(selection.body) if assigns(statement, 'selected_ids'))
    selection_copy = copy.deepcopy(selection)
    selection_copy.body = selection_copy.body[:final + 1]
    statements = parent.body[start:parent.body.index(selection)] + [selection_copy]
    return compile(ast.Module(body=statements, type_ignores=[]), str(path), 'exec')


def camera_ids(reference_root, poses, chunk, chunk_frames, sink_frames, recent_frames, retrieval_frames):
    namespace = dict(torch=torch, c2ws_infer_abs=poses, chunk_id=chunk, chunk_size=chunk_frames,
                     sink_chunks=(sink_frames + chunk_frames - 1) // chunk_frames,
                     recent_chunks=max(1, math.ceil((recent_frames - chunk_frames) / chunk_frames)),
                     num_retrieve_chunks=retrieval_frames // chunk_frames,
                     kv_bank_k=[range(chunk)], selected_ids=[])
    exec(camera_selection_code(reference_root), namespace)
    return namespace['selected_ids']
