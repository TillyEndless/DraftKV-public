"""Explicit bounded checkpoint observations; never a production consumer fallback."""
from contextlib import contextmanager
import json
import os
from pathlib import Path

from .contracts import BackboneError, QKVContext
from .planning import sha256
from draftkv.runtime.config import json_sha256


class AcceptanceStop(BaseException):
    """Only the dedicated worker acceptance handler may consume this sentinel."""
    def __init__(self, record):
        self.record = record


def validate_acceptance(plan, spec):
    options = plan.get("bounded_acceptance")
    if options is None:
        return None
    if (not isinstance(options, dict) or set(options) != {"mode", "start_frame", "layer", "max_forwards"}
            or options["mode"] not in ("capture", "capture_resident", "forward", "probe")
            or any(type(options[k]) is not int for k in ("start_frame", "layer", "max_forwards"))
            or ((options["start_frame"] != 0) if options["mode"] == "capture_resident" else
                (options["start_frame"] < spec.window or options["start_frame"] % spec.chunk))
            or options["start_frame"] >= plan["request"]["latent_frames"]
            or not 0 <= options["layer"] < spec.layers or not 1 <= options["max_forwards"] <= 1000):
        raise BackboneError("BLOCKED_ACCEPTANCE", "explicit mature chunk/layer and bounded forward count required")
    expected = (plan["algorithm"] if options["mode"] == "probe" else
                "native" if options["mode"] in ("capture", "capture_resident") else "draftkv")
    if (expected not in ("native", "draftkv", "shadowkv", "infinigen", "infllm", "freekv", "worldkv", "worldtrace-landmark", "deep-forcing", "future-forcing")
            or (options["mode"] == "probe" and expected == "native")
            or plan["algorithm"] != expected or (expected != "native" and
            (plan.get("algorithm_plugin") or {}).get("entry") != f"src/draftkv/algorithms/{expected.replace('-', '_')}/entry.py")):
        raise BackboneError("BLOCKED_ACCEPTANCE", "capture requires Native; forward/probe require the selected canonical plugin")
    return options


def target(options, context):
    return context.start_frame == options["start_frame"] and context.layer == options["layer"]


def _importance_config(record):
    from draftkv.retrieval.draftattention2_importance import importance_config_from_dict, load_importance_config
    embedded = record.get("importance_scorer") if isinstance(record, dict) else None
    if isinstance(embedded, dict):
        config = importance_config_from_dict(embedded)
        return config, None, record.get("importance_config_sha256")
    root = Path(record.get("repo_root", Path(__file__).resolve().parents[3]))
    settings = ((record.get("algorithm_plugin") or {}).get("config") or {})
    config_path = Path(settings.get(
        "importance_config", "configs/algorithms/draftkv/scorer.json"))
    if not config_path.is_absolute():
        config_path = root / config_path
    config_path = config_path.resolve()
    return load_importance_config(config_path), config_path, sha256(config_path)


def native_capture_geometry(plan):
    """Describe the locked LingBot resizer; leave every generation argument intact."""
    if plan['backbone'] == 'SolarWM33B':
        from .solarwm_geometry import POSITION_POLICY
        return dict(kind='SOURCE_DERIVED_NATIVE_H3_NOT_GPU_MEASUREMENT',
                    native_output_hw=[768, 1344], frozen_output_hw=[768, 1344],
                    latent_hw=[48, 84], attention_grid_hw=[24, 42], frame_tokens=1008,
                    position_policy=POSITION_POLICY, output_spec_status='MATCH')
    if plan['backbone'] not in ('LingBot1.3B', 'LingBot14B'):
        return None
    import numpy as np
    from PIL import Image
    request = plan['request']
    image = Path(request['image'])
    if sha256(image) != request['asset_sha256']['image']:
        raise BackboneError('BLOCKED_INPUT_PROTOCOL', 'Capture image bytes changed')
    source_key = str((Path(plan['native_root']) / 'wan/image2video.py').relative_to(plan['source_root']))
    source = Path(plan['source_root']) / source_key
    if sha256(source) != plan['source_files'][source_key]:
        raise BackboneError('SOURCE_ABI_DRIFT', 'Native spatial resizer source changed')
    with Image.open(image) as img:
        width, height = img.size
    area = request['height'] * request['width']
    aspect = height / width
    # These are the author's NumPy/floor expressions, including their actual
    # floating-point boundary behavior. No requested dimensions are changed.
    latent_h = round(np.sqrt(area * aspect) // 8 // 2 * 2)
    latent_w = round(np.sqrt(area / aspect) // 8 // 2 * 2)
    actual = [latent_h * 8, latent_w * 8]
    frozen = [request['output_height'], request['output_width']]
    return dict(kind='SOURCE_DERIVED_NATIVE_RESIZER_NOT_GPU_MEASUREMENT',
        source_file=source_key, source_sha256=plan['source_files'][source_key],
        image_sha256=request['asset_sha256']['image'], image_hw=[height, width],
        max_area=area, numpy_version=np.__version__, native_output_hw=actual,
        frozen_output_hw=frozen, latent_hw=[latent_h, latent_w],
        attention_grid_hw=[latent_h // 2, latent_w // 2], frame_tokens=latent_h * latent_w // 4,
        output_spec_status='MATCH' if actual == frozen else 'NATIVE_OUTPUT_SPEC_MISMATCH')


def tensor_metadata(tensor):
    """Observe the live tensor before detach, transfer or layout normalization.

    Pointers identify storage only within this worker and are never portable
    addresses. CPU copies must not be used to infer a CUDA source's layout.
    """
    import torch
    device = tensor.device
    device_record = dict(type=device.type, index=device.index, uuid=None)
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        uuid = getattr(properties, "uuid", None)
        if uuid is None:
            raise BackboneError("BLOCKED_ACCEPTANCE", "actual CUDA tensor device UUID unavailable")
        if isinstance(uuid, bytes):
            uuid = uuid.decode("ascii")
        device_record.update(uuid=str(uuid), name=properties.name)
    storage = tensor.untyped_storage()
    return dict(shape=list(tensor.shape), stride=list(tensor.stride()),
                storage_offset=tensor.storage_offset(), storage_nbytes=storage.nbytes(),
                storage_data_ptr=storage.data_ptr(), tensor_data_ptr=tensor.data_ptr(),
                element_size=tensor.element_size(), numel=tensor.numel(),
                dtype=str(tensor.dtype), layout=str(tensor.layout),
                contiguous=tensor.is_contiguous(), pinned=tensor.is_pinned(),
                device=device_record, observed_before_cpu_copy=True)


def live_capture_metadata(plan, context, bank, selected, *, history_source_metadata=None):
    """Keep the native seam and historical source allocations separate."""
    import torch.distributed as dist
    from .catalog import MODELS
    rank, sp = int(os.environ.get("RANK", "0")), plan["sp"]
    if sp > 1 and (not dist.is_initialized() or dist.get_world_size() != sp or dist.get_rank() != rank):
        raise BackboneError("BLOCKED_ACCEPTANCE", "actual native SP rank group differs from capture plan")
    local_heads = context.query.shape[2]
    global_heads = MODELS[plan["backbone"]].heads
    sp_mode = plan.get("sp_mode", "single")
    if not 0 <= rank < sp:
        raise BackboneError("BLOCKED_ACCEPTANCE", "native rank differs from the model topology")
    if sp_mode == "query_sequence_replicated_kv":
        if local_heads != global_heads:
            raise BackboneError("BLOCKED_ACCEPTANCE", "replicated-KV SP must retain every native model head")
        local_head_ids = list(range(global_heads))
        source_contract = "exact query-sequence shard; full-head/full-token K/V replicated on each rank"
    else:
        if local_heads * sp != global_heads:
            raise BackboneError("BLOCKED_ACCEPTANCE", "native head-shard layout differs from the model")
        local_head_ids = list(range(rank * local_heads, (rank + 1) * local_heads))
        source_contract = ("native all_to_all scatter_dim=2/gather_dim=1; contiguous head chunks"
                           if sp > 1 else "native unsharded heads")
    tensors = {name: tensor_metadata(getattr(context, attr)) for name, attr in (
        ("query", "query"), ("current_key", "key"), ("current_value", "value"),
        ("native_keys", "native_keys"), ("native_values", "native_values"),
        ("raw_query", "raw_query"), ("raw_key", "raw_key"))}
    if getattr(context, 'fixed_prefix', None) is not None:
        tensors.update({name: tensor_metadata(tensor) for name, tensor in
                        zip(('fixed_prefix_key', 'fixed_prefix_value'), context.fixed_prefix)})
    devices = {item["device"]["uuid"] for item in tensors.values() if item["device"]["type"] == "cuda"}
    if context.query.device.type == "cuda" and (len(devices) != 1 or
            any(item["device"]["type"] != "cuda" for item in tensors.values())):
        raise BackboneError("BLOCKED_ACCEPTANCE", "native QKV seam spans unexpected devices")
    history = []
    for fid in selected:
        source = (history_source_metadata or {}).get(fid)
        history.append(dict(frame_id=fid, bank_key=tensor_metadata(bank[fid][0]),
                            bank_value=tensor_metadata(bank[fid][1]), clean_commit_source=source))
    return dict(schema_version=1, observation="LIVE_NATIVE_SEAM_BEFORE_CPU_COPY",
                tensors=tensors, historical_pages=history,
                head_mapping=dict(axis=2, rank=rank, sp=sp, global_heads=global_heads,
                    local_heads=local_heads, local_head_ids=local_head_ids,
                    source_contract=source_contract, sp_mode=sp_mode,
                    head_padding=False, independent_head_value_golden=False))


def _complete_history_inputs(plan, context, bank, descriptors, sources, evidence, native_metadata):
    """Validate the complete target-layer bank before writing any artifact."""
    import torch
    from draftkv.retrieval.draftattention2_importance import regions_per_frame
    frames = tuple(range(context.start_frame))
    if any(not isinstance(mapping, dict) or any(type(fid) is not int for fid in mapping)
           or set(mapping) != set(frames) for mapping in (bank, descriptors, sources)):
        raise BackboneError("BLOCKED_HISTORY_CAPTURE", "bank/descriptors/clean sources must exactly cover all prior absolute frames")
    h, w = int(context.grid_sizes[0][1]), int(context.grid_sizes[0][2])
    heads, dim, tokens = context.query.shape[2], context.query.shape[3], context.frame_tokens
    if h * w != tokens or context.raw_query.shape != context.query.shape or context.raw_key.shape != context.key.shape:
        raise BackboneError("BLOCKED_HISTORY_CAPTURE", "original raw Q/K and native spatial geometry differ")
    importance, _, _ = _importance_config(plan)
    regions = regions_per_frame(h, w, importance.capacity)
    uuid = native_metadata['tensors']['query']['device'].get('uuid')
    if not uuid or native_metadata['tensors']['query']['device'].get('type') != 'cuda':
        raise BackboneError("BLOCKED_HISTORY_CAPTURE", "complete Native history requires an actual CUDA source identity")
    candidate_ids = tuple(fid for fid in frames if fid >= context.sink_frames)
    query_regions = (context.query.shape[1] // tokens) * regions
    shapes = dict(local_frame_mass=(1, heads, query_regions, len(candidate_ids)),
                  global_frame_mass=(1, heads * plan['sp'], query_regions, len(candidate_ids)),
                  reduced_frame_scores=(1, len(candidate_ids)))
    if (not isinstance(evidence, dict) or tuple(evidence.get('candidate_frame_ids', ())) != candidate_ids
            or any(name not in evidence or not isinstance(evidence[name], torch.Tensor)
                   or tuple(evidence[name].shape) != shape or evidence[name].dtype != torch.float32
                   for name, shape in shapes.items())):
        raise BackboneError("BLOCKED_HISTORY_CAPTURE", "actual local/all-rank masses and reduced selection scores are required")
    for raw in (context.raw_query, context.raw_key):
        if raw.dtype != torch.bfloat16 or raw.ndim != 4:
            raise BackboneError("BLOCKED_HISTORY_CAPTURE", "original BF16 raw Q/K are required")
    chunk_forward = {}
    for fid in frames:
        pair, descriptor, source = bank[fid], descriptors[fid], sources[fid]
        if (not isinstance(pair, (list, tuple)) or len(pair) != 2
                or any(not isinstance(t, torch.Tensor) or t.device.type != 'cpu' or t.dtype != torch.bfloat16
                       or tuple(t.shape) != (tokens, heads, dim) for t in pair)
                or not isinstance(descriptor, torch.Tensor) or descriptor.dtype != torch.float16
                or tuple(descriptor.shape) != (2, 1, heads, regions, dim)):
            raise BackboneError("BLOCKED_HISTORY_CAPTURE", "full-resolution CPU KV or post-RoPE Spatial8x16 mean/max descriptor differs")
        if (not isinstance(source, dict) or source.get('clean') is not True
                or source.get('publish_after_complete_forward') is not True
                or type(source.get('forward_id')) is not int or source['forward_id'] < 1
                or type(source.get('start_frame')) is not int
                or source['start_frame'] != fid // plan['native_chunk_frames'] * plan['native_chunk_frames']):
            raise BackboneError("BLOCKED_HISTORY_CAPTURE", "history source is not a completed clean Native forward")
        prior = chunk_forward.setdefault(source['start_frame'], source['forward_id'])
        if prior != source['forward_id']:
            raise BackboneError("BLOCKED_HISTORY_CAPTURE", "a historical chunk mixes different clean forwards")
        for name in ('key', 'value'):
            meta = source.get(name, {})
            if (meta.get('device', {}).get('type') != 'cuda' or meta['device'].get('uuid') != uuid
                    or meta.get('observed_before_cpu_copy') is not True
                    or meta.get('shape') != [tokens, heads, dim] or meta.get('dtype') != 'torch.bfloat16'
                    or type(meta.get('storage_nbytes')) is not int or meta['storage_nbytes'] < tokens * heads * dim * 2
                    or not isinstance(meta.get('stride'), list) or len(meta['stride']) != 3):
                raise BackboneError("BLOCKED_HISTORY_CAPTURE", "clean page source lacks exact original CUDA allocation identity")
    return frames, candidate_ids, (h, w)


def save_complete_history(plan, context, bank, descriptors, sources, evidence, native_metadata, selection_trace):
    """Archive real target-layer history and already-computed selection inputs.

    No selection/attention is computed here. Existing local/all-gather/reduced
    tensors are copied exactly; no additional distributed collective occurs.
    """
    import torch
    from .catalog import MODELS
    importance, _, importance_sha = _importance_config(plan)
    options = {**plan, 'native_chunk_frames': MODELS[plan['backbone']].chunk,
               'importance_scorer': importance.to_dict(),
               'importance_config_sha256': importance_sha}
    frames, candidates, grid = _complete_history_inputs(options, context, bank, descriptors, sources, evidence, native_metadata)
    if (not isinstance(selection_trace, dict) or selection_trace.get('policy') != 'draftkv_hierarchical_v1_fast'
            or selection_trace.get('candidate_ids') != list(candidates)
            or selection_trace.get('query_start') != context.start_frame
            or type(selection_trace.get('frame_budget')) is not int
            or not 1 <= selection_trace['frame_budget'] <= len(candidates)
            or any(not isinstance(selection_trace.get(name), list) for name in
                   ('selected_absolute_frame_ids', 'local', 'salient', 'protected'))):
        raise BackboneError('BLOCKED_HISTORY_CAPTURE', 'actual DraftAttention2-score/Hierarchical-V1 selection trace required')
    descriptor_sources = {fid: tensor_metadata(descriptors[fid]) for fid in frames}
    evidence_sources = {name: tensor_metadata(evidence[name]) for name in
                        ('local_frame_mass', 'global_frame_mass', 'reduced_frame_scores')}
    uuid = native_metadata['tensors']['query']['device']['uuid']
    if any(meta['device'].get('type') != 'cuda' or meta['device'].get('uuid') != uuid
           for meta in [*descriptor_sources.values(), *evidence_sources.values()]):
        raise BackboneError('BLOCKED_HISTORY_CAPTURE', 'Draft descriptors/masses must originate on the assigned Native CUDA rank')
    def snapshot(tensor):
        return tensor.detach().to('cpu').contiguous().clone()
    rank = native_metadata['head_mapping']['rank']
    history = {fid: dict(key=snapshot(bank[fid][0]), value=snapshot(bank[fid][1]),
                        descriptor=snapshot(descriptors[fid]), clean_commit_source=sources[fid],
                        descriptor_source=descriptor_sources[fid]) for fid in frames}
    payload = dict(schema_version=1, input_kind='REAL_NATIVE_COMPLETE_CLEAN_HISTORY',
        scope='target attention layer only; all earlier absolute frames; not every model layer',
        rank=rank, sp=plan['sp'], backbone=plan['backbone'], layer=context.layer,
        layers=MODELS[plan['backbone']].layers, start_frame=context.start_frame,
        native_chunk_frames=options['native_chunk_frames'], frame_tokens=context.frame_tokens,
        sink_frame_ids=list(range(context.sink_frames)), frame_ids=list(frames), candidate_frame_ids=list(candidates),
        checkpoint_sha256=plan['checkpoint_sha256'], model_source_sha256=json_sha256(plan['source_files']),
        case_id=plan['request']['case_id'], history=history, grid_hw=list(grid),
        importance_scorer=importance.to_dict(), importance_config_sha256=importance_sha,
        raw_query=snapshot(context.raw_query), raw_key=snapshot(context.raw_key),
        raw_query_source=native_metadata['tensors']['raw_query'], raw_key_source=native_metadata['tensors']['raw_key'],
        selection_evidence={name: snapshot(evidence[name]) for name in
            ('local_frame_mass', 'global_frame_mass', 'reduced_frame_scores')},
        selection_evidence_sources=evidence_sources,
        selection_trace=selection_trace, native_absolute_rope=True,
        independently_recomputed=False, golden=False, forward_complete=False)
    folder = Path(plan['run_dir']) / 'acceptance'
    folder.mkdir(exist_ok=True)
    path = folder / f'clean_history_rank{rank}.pt'
    with path.open('xb') as handle:
        torch.save(payload, handle)
    return dict(path=str(path.resolve()), sha256=sha256(path),
                input_kind=payload['input_kind'], scope=payload['scope'])


def load_complete_history(capture):
    """Hash-check and validate a complete history artifact on CPU, fail closed."""
    import torch
    reference = capture.get('full_clean_history_artifact')
    if not isinstance(reference, dict) or not isinstance(reference.get('path'), str):
        raise BackboneError('BLOCKED_HISTORY_CAPTURE', 'complete history artifact missing; partial legacy capture cannot verify V1 IDs')
    path = Path(reference['path'])
    if not path.is_file() or sha256(path) != reference.get('sha256'):
        raise BackboneError('BLOCKED_HISTORY_CAPTURE', 'complete history artifact SHA mismatch')
    data = torch.load(path, map_location='cpu', weights_only=True)
    if (not isinstance(data, dict) or data.get('input_kind') != 'REAL_NATIVE_COMPLETE_CLEAN_HISTORY'
            or data.get('schema_version') != 1 or any(data.get(name) != capture.get(name) for name in
                ('rank', 'sp', 'layer', 'layers', 'start_frame', 'frame_tokens', 'checkpoint_sha256',
                 'model_source_sha256', 'case_id', 'sink_frame_ids'))):
        raise BackboneError('BLOCKED_HISTORY_CAPTURE', 'complete history provenance differs from real QKV capture')
    if (data.get('selection_trace') != capture.get('selection_trace')
            or not isinstance(data.get('selection_trace'), dict)
            or data['selection_trace'].get('selected_absolute_frame_ids') != capture.get('selected_frame_ids')
            or data.get('raw_query_source') != capture['native_tensor_metadata']['tensors']['raw_query']
            or data.get('raw_key_source') != capture['native_tensor_metadata']['tensors']['raw_key']
            or data.get('importance_scorer') != capture.get('importance_scorer')
            or data.get('importance_config_sha256') != capture.get('importance_config_sha256')):
        raise BackboneError('BLOCKED_HISTORY_CAPTURE', 'complete history trace/raw source differs from real QKV capture')
    history = data.get('history', {})
    from types import SimpleNamespace
    context = SimpleNamespace(start_frame=data['start_frame'], grid_sizes=[[0, *data['grid_hw']]],
        query=capture['q'], key=capture['q'], raw_query=data['raw_query'], raw_key=data['raw_key'],
        frame_tokens=data['frame_tokens'], sink_frames=len(data['sink_frame_ids']))
    evidence = {**data.get('selection_evidence', {}), 'candidate_frame_ids': data.get('candidate_frame_ids')}
    frames, candidates, _ = _complete_history_inputs(data, context,
        {fid: (row['key'], row['value']) for fid, row in history.items()},
        {fid: row['descriptor'] for fid, row in history.items()},
        {fid: row['clean_commit_source'] for fid, row in history.items()}, evidence, capture['native_tensor_metadata'])
    if data.get('frame_ids') != list(frames) or data.get('candidate_frame_ids') != list(candidates):
        raise BackboneError('BLOCKED_HISTORY_CAPTURE', 'complete history absolute frame order changed')
    materialized_ids = capture['sink_frame_ids'] + capture['selected_frame_ids']
    tokens = data['frame_tokens']
    for index, fid in enumerate(materialized_ids):
        for component in ('key', 'value'):
            offset = capture.get('fixed_prefix_tokens', 0) + index * tokens
            actual = capture['k' if component == 'key' else 'v'][0, offset:offset + tokens]
            archived = history[fid][component]
            if not torch.equal(actual.contiguous().view(torch.uint8), archived.contiguous().view(torch.uint8)):
                raise BackboneError('BLOCKED_HISTORY_CAPTURE', 'complete clean history bytes differ from materialized QKV pages')
    if any(t.device.type != 'cpu' for t in (data['raw_query'], data['raw_key'],
            *data['selection_evidence'].values())):
        raise BackboneError('BLOCKED_HISTORY_CAPTURE', 'history verification must consume archived CPU tensors')
    uuid = capture['native_tensor_metadata']['tensors']['query']['device']['uuid']
    metadata = [data.get('raw_query_source', {}), data.get('raw_key_source', {}),
                *[row.get('descriptor_source', {}) for row in history.values()],
                *[data.get('selection_evidence_sources', {}).get(name, {}) for name in
                  ('local_frame_mass', 'global_frame_mass', 'reduced_frame_scores')]]
    if any(meta.get('device', {}).get('type') != 'cuda' or meta['device'].get('uuid') != uuid
           or meta.get('observed_before_cpu_copy') is not True for meta in metadata):
        raise BackboneError('BLOCKED_HISTORY_CAPTURE', 'complete history lost original descriptor/rawQ/mass device provenance')
    return data


def recompute_complete_history_v1(captures):
    """Independently recompute Hierarchical-V1 IDs from ALL ranks' archived post-RoPE Q/descriptors.

    CPU/GPU FP32 reductions may differ in low bits. Report their errors without
    inventing a numerical tolerance; selection IDs/local/salient must match
    exactly. This establishes ID replay, never CUDA Golden or model Forward.
    """
    import torch
    from draftkv.retrieval import reduce_frame_mass, select_v1
    from draftkv.retrieval.draftattention2_importance import frame_mass
    if not isinstance(captures, (list, tuple)) or not captures:
        raise BackboneError('BLOCKED_HISTORY_CAPTURE', 'every actual rank capture is required')
    sp = captures[0]['sp']
    if len(captures) != sp or [c.get('rank') for c in captures] != list(range(sp)):
        raise BackboneError('BLOCKED_HISTORY_CAPTURE', 'complete ordered SP world required; no rank replication')
    artifacts = [load_complete_history(capture) for capture in captures]
    first = artifacts[0]
    for data in artifacts:
        if any(data[key] != first[key] for key in ('sp', 'layer', 'start_frame', 'case_id', 'checkpoint_sha256',
                'model_source_sha256', 'candidate_frame_ids', 'grid_hw', 'frame_tokens', 'importance_scorer')):
            raise BackboneError('BLOCKED_HISTORY_CAPTURE', 'rank history provenance or candidate order differs')
        if any(data['selection_trace'].get(key) != first['selection_trace'].get(key) for key in
               ('query_start', 'frame_budget', 'candidate_ids', 'selected_absolute_frame_ids',
                'chronological', 'local', 'salient', 'protected')):
            raise BackboneError('BLOCKED_HISTORY_CAPTURE', 'rank history V1 decisions differ')
    local_masses = [frame_mass(
        capture['q'], [data['history'][fid]['descriptor'] for fid in data['candidate_frame_ids']],
        *data['grid_hw'], config=_importance_config(data)[0])
        for capture, data in zip(captures, artifacts)]
    actual_global = torch.cat([data['selection_evidence']['local_frame_mass'] for data in artifacts], dim=1)
    if any(not torch.equal(actual_global, data['selection_evidence']['global_frame_mass']) for data in artifacts):
        raise BackboneError('BLOCKED_HISTORY_CAPTURE', 'archived all-gather masses do not match every actual rank shard')
    scores = reduce_frame_mass(torch.cat(local_masses, dim=1))
    trace = first['selection_trace']
    decision = select_v1(scores, first['candidate_frame_ids'], trace['frame_budget'],
                         query_start=first['start_frame'], sink_ids=first['sink_frame_ids'],
                         protect_recent=len(trace['protected']))
    for capture, data in zip(captures, artifacts):
        saved = data['selection_trace']
        original = select_v1(data['selection_evidence']['reduced_frame_scores'], data['candidate_frame_ids'],
            saved['frame_budget'], query_start=data['start_frame'], sink_ids=data['sink_frame_ids'],
            protect_recent=len(saved['protected']))
        if (list(decision.chronological) != capture['selected_frame_ids'] or original.ranked != decision.ranked
                or any(list(getattr(decision, key)) != saved[key]
                       or list(getattr(original, key)) != saved[key]
                       for key in ('chronological', 'local', 'salient', 'protected'))):
            raise BackboneError('BLOCKED_HISTORY_CAPTURE', 'independent complete-history V1 IDs differ')
    return dict(status='COMPLETE_HISTORY_D2_POSTROPE_HIER_V1_IDS_RECOMPUTED_MATCH', sp=sp,
        selected_frame_ids=list(decision.chronological), importance_ranked_frame_ids=list(decision.ranked),
        local_ids=list(decision.local), salient_ids=list(decision.salient),
        local_mass_max_abs_errors=[(a - d['selection_evidence']['local_frame_mass']).abs().max().item()
                                  for a, d in zip(local_masses, artifacts)],
        score_max_abs_error=(scores - first['selection_evidence']['reduced_frame_scores']).abs().max().item(),
        score_max_abs_errors=[(scores - d['selection_evidence']['reduced_frame_scores']).abs().max().item()
                              for d in artifacts],
        numeric_bitwise_equal=(all(torch.equal(a, d['selection_evidence']['local_frame_mass'])
                                  for a, d in zip(local_masses, artifacts))
                               and all(torch.equal(scores, d['selection_evidence']['reduced_frame_scores'])
                                       for d in artifacts)),
        scope=first['scope'], cuda_golden=False, forward_complete=False)


def save_capture(plan, context, bank, selected, *, mode, output=None, selection_trace=None,
                 history_source_metadata=None, history_descriptors=None, selection_evidence=None,
                 attention_position_policy=None, position_map=None):
    """Materialize exact absolute-ID pages, never uninitialized transfer scratch."""
    import torch
    from .catalog import MODELS
    sink = tuple(range(min(context.sink_frames, context.start_frame)))
    selected = tuple(selected)
    if mode == "capture_resident":
        if context.start_frame != 0 or sink or selected or bank:
            raise BackboneError("BLOCKED_ACCEPTANCE", "zero-history resident Capture cannot include or invent historical KV")
    elif not selected or set(sink) & set(selected) or any(fid not in bank for fid in sink + selected):
        raise BackboneError("BLOCKED_ACCEPTANCE", "mature capture needs actual clean sink and selected history")
    native_metadata = live_capture_metadata(plan, context, bank, sink + selected,
                                            history_source_metadata=history_source_metadata)
    if attention_position_policy is not None:
        native_metadata.update(observation='LIVE_DRAFTKV_CONSUMER_BEFORE_CPU_COPY',
                               attention_position_policy=attention_position_policy)
    full_history = None
    if mode == 'capture' and history_descriptors is not None:
        if (not isinstance(selection_trace, dict)
                or list(selected) != selection_trace.get('chronological')
                or selected != tuple(sorted(selected))):
            raise BackboneError('BLOCKED_HISTORY_CAPTURE', 'capture physical KV order must match chronological V1 selection')
        full_history = save_complete_history(plan, context, bank, history_descriptors,
            history_source_metadata, selection_evidence, native_metadata, selection_trace)
    q = context.query.detach().to("cpu").contiguous()
    kv = []
    prefix = getattr(context, 'fixed_prefix', None)
    prefix_tokens = prefix[0].shape[1] if prefix is not None else 0
    for component, current in enumerate((context.key, context.value)):
        pages = [bank[fid][component].unsqueeze(0) for fid in sink + selected]
        if prefix is not None:
            pages.insert(0, prefix[component].detach().to('cpu'))
        kv.append(torch.cat(pages + [current.detach().to("cpu")], dim=1).contiguous())
    rank, sp = int(os.environ.get("RANK", "0")), plan["sp"]
    source_sha = json_sha256(plan["source_files"])
    global_heads = MODELS[plan["backbone"]].heads
    local_head_ids = (list(range(global_heads)) if plan.get("sp_mode") == "query_sequence_replicated_kv"
                      else list(range(rank * q.shape[2], (rank + 1) * q.shape[2])))
    capture = dict(input_kind="REAL_CHECKPOINT_QKV", input_placement="attention_local", rank=rank, sp=sp,
                   sp_mode=plan.get("sp_mode", "single"), global_heads=global_heads,
                   local_head_ids=local_head_ids,
                   checkpoint_sha256=plan["checkpoint_sha256"], model_source_sha256=source_sha,
                   case_id=plan["request"]["case_id"], q=q, k=kv[0], v=kv[1],
                   frame_tokens=context.frame_tokens, sink_tokens=prefix_tokens + len(sink) * context.frame_tokens,
                   fixed_prefix_tokens=prefix_tokens, backbone=plan['backbone'],
                   native_position_policy=attention_position_policy or getattr(getattr(context, 'native_geometry', None), 'policy', 'native_absolute_rope'),
                   attention_position_policy=attention_position_policy,
                   attention_position_map=position_map,
                   selected_frame_ids=list(selected), sink_frame_ids=list(sink),
                   current_frame_ids=list(range(context.start_frame, context.start_frame + context.current_frames)),
                   layer=context.layer, layers=MODELS[plan["backbone"]].layers,
                   start_frame=context.start_frame, native_absolute_rope=attention_position_policy is None,
                   importance_scorer=_importance_config(plan)[0].to_dict(),
                   importance_config_sha256=_importance_config(plan)[2],
                   selection_trace=selection_trace, baseline_algorithm=plan["algorithm"],
                   native_tensor_metadata=native_metadata,
                   native_input_geometry=plan.get('native_capture_geometry'),
                   full_clean_history_artifact=full_history,
                   observation_scope=("NATIVE_BASELINE_ZERO_HISTORY_RESIDENT_CAPTURE_ONLY" if mode == "capture_resident"
                       else "NATIVE_BASELINE_D2_POSTROPE_HIER_V1_SELECTION_CAPTURE_ONLY" if mode == "capture"
                       else "CANONICAL_B01_BOUNDED_ATTENTION"),
                   b01_validated=False, b01_consumer_observed=mode == "forward", source_files=plan["source_files"])
    if getattr(getattr(context, "native_geometry", None), "positions", None) is not None:
        capture["native_temporal_origin"] = float(context.native_geometry.positions[0, 0])
    folder = Path(plan["run_dir"]) / "acceptance"
    folder.mkdir(exist_ok=True)
    path = folder / f"qkv_rank{rank}.pt"
    with path.open("xb") as handle:
        torch.save(capture, handle)
    record = dict(rank=rank, path=str(path.resolve()), sha256=sha256(path),
                  checkpoint_sha256=plan["checkpoint_sha256"], case_id=capture["case_id"], model_source_sha256=source_sha)
    receipt = dict(schema_version=1, mode=mode, status="CAPTURE_ONLY_NO_B01_VALIDATION" if mode in ("capture", "capture_resident") else "BOUNDED_B01_ATTENTION_OBSERVED",
                   rank=rank, sp=sp, capture=record, layer=context.layer, start_frame=context.start_frame,
                   query_shape=list(q.shape), kv_shape=list(kv[0].shape), selected_frame_ids=list(selected),
                   native_tensor_metadata=native_metadata,
                   native_input_geometry=plan.get('native_capture_geometry'),
                   full_clean_history_artifact=full_history,
                   importance_scorer=_importance_config(plan)[0].to_dict(),
                   attention_position_policy=attention_position_policy,
                   attention_position_map=position_map,
                   importance_config_sha256=_importance_config(plan)[2],
                   output_shape=list(output.shape) if output is not None else None,
                   scope="model prefix through target attention; no completed target projection/whole-model-forward claim",
                   forward_complete=False, clean_commit_verified=False,
                   golden=False, video_e2e=False, official_evaluation=False)
    with (folder / f"rank{rank}.json").open("x") as handle:
        json.dump(receipt, handle, indent=2)
    raise AcceptanceStop(receipt)


class NativeCaptureSession:
    """Native FA2 trajectory plus exact D2-importance/Hierarchical-V1 observation.

    Does not instantiate a DraftKV hook, native binding, ProducerPlan or B01.
    The observer banks native clean current K/V only after a full model forward.
    """
    native_observer = True

    def __init__(self, core, options):
        self.core, self.options = core, options
        self.bank, self.descriptors = {}, {}
        self.bank_source_metadata, self.pending_source_metadata = {}, {}
        self.active, self.pending, self.seen = None, {}, set()
        self.forward_sequence = 0
        self.importance, self.importance_path, self.importance_sha256 = _importance_config(core.plan)

    @contextmanager
    def forward(self, *, start_frame, clean, timestep):
        self.forward_sequence += 1
        if self.forward_sequence > self.options["max_forwards"] or start_frame > self.options["start_frame"]:
            raise BackboneError("BLOCKED_ACCEPTANCE", "bounded Native observer missed its requested target")
        self.active = dict(start_frame=start_frame, clean=clean, timestep=timestep)
        self.pending, self.seen = {}, set()
        self.pending_source_metadata = {}
        try:
            yield
            if self.seen != set(range(self.core.spec.layers)):
                raise BackboneError("PARTIAL_NATIVE_WIRING", "Native observer did not complete all model layers")
            if clean:
                for layer, (pages, descriptors) in self.pending.items():
                    self.bank.setdefault(layer, {}).update(pages)
                    self.descriptors.setdefault(layer, {}).update(descriptors)
                    self.bank_source_metadata.setdefault(layer, {}).update(self.pending_source_metadata.get(layer, {}))
        finally:
            self.active, self.pending = None, {}
            self.pending_source_metadata = {}

    def attend(self, layer, query, key, value, nk, nv, raw_query, raw_key,
               tokens, grid, freqs, native_fa2, native_cache=None, *, fixed_prefix=None, native_geometry=None):
        import torch
        from draftkv.retrieval.draftattention2_importance import descriptor, frame_mass
        from draftkv.retrieval.hierarchical_v1 import reduce_frame_mass
        from draftkv.retrieval import SelectionContext, select_v1
        spec, plan = self.core.spec, self.core.plan
        if self.active is None or layer in self.seen or key.shape[1] % tokens:
            raise BackboneError("PARTIAL_NATIVE_WIRING", "Native capture requires unique whole-frame layer attention")
        semantic_sp = 1 if plan.get("sp_mode") == "query_sequence_replicated_kv" else plan["sp"]
        context = QKVContext(layer, query, key, value, raw_query, raw_key, nk, nv,
                             self.active["start_frame"], tokens, key.shape[1] // tokens,
                             spec.sink, spec.window, semantic_sp, grid, freqs, plan["request"],
                             self.active["clean"], self.active["timestep"], native_cache, self.forward_sequence,
                             fixed_prefix, native_geometry)
        h, w = int(grid[0][1]), int(grid[0][2])
        geometry = plan.get('native_capture_geometry')
        expected_tokens = geometry['frame_tokens'] if geometry else plan['history_plan']['frame_tokens']
        from .sequence_parallel import local_heads
        expected_heads = local_heads(spec.heads, spec.family, plan["sp"])
        if (h * w != tokens or query.shape != key.shape or key.shape != value.shape
                or query.dtype != torch.bfloat16 or query.shape[2:] != (expected_heads, spec.head_dim)
                or query.shape[0] != 1 or tokens != expected_tokens
                or (geometry and [h, w] != geometry['attention_grid_hw'])):
            observed = dict(layer=layer, start_frame=context.start_frame, frame_tokens=tokens,
                grid_hw=[h, w], tensors={name: tensor_metadata(tensor) for name, tensor in
                    (('query', query), ('key', key), ('value', value), ('raw_query', raw_query), ('raw_key', raw_key))},
                expected_frame_tokens=expected_tokens, source_derived_geometry=geometry)
            self.core.trace('native_capture_geometry_rejected', **observed)
            raise BackboneError("BLOCKED_ACCEPTANCE", "actual native geometry differs: " + json.dumps(observed))
        if target(self.options, context):
            bank = self.bank.get(layer, {})
            if self.options["mode"] == "capture_resident":
                if context.start_frame != 0 or bank or context.current_frames < 1:
                    raise BackboneError("BLOCKED_ACCEPTANCE", "resident capture requires genuine first-chunk empty history")
                save_capture(plan, context, bank, (), mode="capture_resident")
            if set(bank) != set(range(context.start_frame)):
                raise BackboneError("DIRTY_HISTORY", "capture requires the complete native clean historical frontier")
            candidates = tuple(fid for fid in bank if spec.sink <= fid < context.start_frame)
            budget = min(spec.window - spec.sink - spec.chunk, len(candidates))
            masses = frame_mass(query, [self.descriptors[layer][fid] for fid in candidates],
                                h, w, config=self.importance)
            local_masses = masses
            if plan.get("sp_mode") == "native_head_shard" and plan["sp"] > 1:
                import torch.distributed as dist
                if not dist.is_initialized() or dist.get_world_size() != plan["sp"]:
                    raise BackboneError("BLOCKED_ACCEPTANCE", "capture requires the original native SP head group")
                shards = [torch.empty_like(masses) for _ in range(plan["sp"])]
                dist.all_gather(shards, masses)
                masses = torch.cat(shards, dim=1)
            scores = reduce_frame_mass(masses)
            from draftkv.algorithms.draftkv.backbone import (
                canonical_protect_recent, CANONICAL_SALIENT_Z, CANONICAL_RELATIVE_FLOOR)
            decision = select_v1(scores, candidates, budget, query_start=context.start_frame,
                                 sink_ids=tuple(range(spec.sink)),
                                 protect_recent=canonical_protect_recent(budget),
                                 salient_z=CANONICAL_SALIENT_Z,
                                 relative_floor=CANONICAL_RELATIVE_FLOOR,
                                 context=SelectionContext(layer=layer, chunk=context.start_frame))
            save_capture(plan, context, bank, decision.chronological, mode="capture", selection_trace=decision.trace.to_dict(),
                         history_source_metadata=self.bank_source_metadata.get(layer),
                         history_descriptors=self.descriptors.get(layer),
                         selection_evidence=dict(candidate_frame_ids=list(candidates), local_frame_mass=local_masses,
                                                 global_frame_mass=masses, reduced_frame_scores=scores))
        result = native_fa2(query, nk, nv)
        if context.clean:
            pages, descriptors, source_metadata = {}, {}, {}
            for rank in range(context.current_frames):
                fid = context.start_frame + rank
                a, b = rank * tokens, (rank + 1) * tokens
                source_metadata[fid] = dict(key=tensor_metadata(key[0, a:b]), value=tensor_metadata(value[0, a:b]),
                                           forward_id=self.forward_sequence, start_frame=context.start_frame,
                                           clean=True, publish_after_complete_forward=True)
                pages[fid] = tuple(t[0, a:b].detach().to("cpu").contiguous().clone() for t in (key, value))
                descriptors[fid] = descriptor(
                    key[:, a:b], h, w, capacity=self.importance.capacity).detach().clone()
            self.pending[layer] = pages, descriptors
            self.pending_source_metadata[layer] = source_metadata
        self.seen.add(layer)
        return result


class ForwardObserver:
    """Wrap an explicitly selected canonical hook without changing its consumer."""
    def __init__(self, hook, plan, options):
        self.hook, self.owner, self.plan, self.options = hook, hook.owner, plan, options

    def route(self, context):
        return self.hook.route(context)

    def consume(self, context, selection, native_fa2):
        if context.forward_id > self.options["max_forwards"] or context.start_frame > self.options["start_frame"]:
            raise BackboneError("BLOCKED_ACCEPTANCE", "bounded canonical forward missed its requested target")
        output = self.hook.consume(context, selection, native_fa2)
        if target(self.options, context):
            decision = selection.metadata.get("draftkv_plan", {}).get("decision")
            from dataclasses import replace
            pending = selection.metadata['draftkv_plan']
            current_tokens = context.current_frames * context.frame_tokens
            actual = replace(context, query=selection.query,
                             key=selection.keys[:, -current_tokens:],
                             value=selection.values[:, -current_tokens:],
                             fixed_prefix=pending['fixed_prefix'])
            actual_bank = {**self.owner.committed[context.layer], **pending['consumer_bank']}
            save_capture(self.plan, actual, actual_bank,
                         selection.metadata["selected_frame_ids"], mode="forward", output=output,
                         selection_trace=decision.trace.to_dict() if decision else None,
                         attention_position_policy=pending['position']['position_policy'],
                         position_map=pending['position'])
        return output

    def after_attention(self, context, selection):
        return self.hook.after_attention(context, selection)


class AttentionProbe:
    """Optional bounded observation of any shared canonical method, never a gate.

    The actual hook and consumer execute unchanged. Stop after a mature target
    attention; previous chunks completed clean transactions, the target forward
    is intentionally incomplete and is rolled back by HookSession.
    """
    def __init__(self, hook, plan, options):
        self.hook, self.owner, self.plan, self.options = hook, hook.owner, plan, options

    def __getattr__(self, name):
        return getattr(self.hook, name)

    def consume(self, context, selection, native_fa2):
        return self.hook.consume(context, selection, native_fa2)

    def observe_attention_result(self, context, selection, output, calls):
        # HookSession owns ordinary Native FA2 dispatch, so observation belongs
        # after that dispatch as well as after custom query/head/B01 consumers.
        if context.forward_id > self.options["max_forwards"] or context.start_frame > self.options["start_frame"]:
            raise BackboneError("BLOCKED_ACCEPTANCE", "bounded attention probe missed its target")
        if not target(self.options, context):
            return
        import torch
        algorithm = self.plan['algorithm']
        if not selection.retrieved_ids or (calls == 0) != (algorithm == 'draftkv'):
            raise BackboneError("BLOCKED_ACCEPTANCE", "probe requires real retrieval and the declared consumer")
        if output.shape != context.query.shape or not bool(torch.isfinite(output).all()):
            raise BackboneError("BLOCKED_ACCEPTANCE", "invalid actual attention output")
        if (self.owner.frontier != context.start_frame if algorithm == "worldtrace-landmark"
                else tuple(self.owner.history_ids(context.layer)) != tuple(range(context.start_frame))):
            raise BackboneError("DIRTY_HISTORY", "probe requires all preceding clean chunks")
        record = dict(schema_version=1, mode='probe', status='BOUNDED_CANONICAL_ATTENTION_OBSERVED',
            algorithm=algorithm, rank=int(os.environ.get('RANK', '0')), sp=self.plan['sp'],
            checkpoint_sha256=self.plan['checkpoint_sha256'], case_id=self.plan['request']['case_id'],
            model_source_sha256=json_sha256(self.plan['source_files']),
            plugin_source_sha256=json_sha256(self.plan['algorithm_plugin']['files']),
            layer=context.layer, start_frame=context.start_frame, query_shape=list(context.query.shape),
            output_shape=list(output.shape), output_finite=True, native_fa2_calls=calls,
            consumer='streaming_fa2' if algorithm == 'draftkv' else 'native_fa2',
            retrieved_ids=list(selection.retrieved_ids), current_ids=list(selection.current_ids),
            effective_budget=self.plan['effective_budget'],
            clean_history_frontier=context.start_frame, forward_complete=False,
            video_e2e=False, official_evaluation=False,
            scope='real checkpoint prefix through mature target attention; target forward intentionally incomplete')
        folder = Path(self.plan['run_dir']) / 'acceptance'
        folder.mkdir(exist_ok=True)
        with (folder / f"rank{record['rank']}.json").open('x') as stream:
            json.dump(record, stream, indent=2)
        raise AcceptanceStop(record)
