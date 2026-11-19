"""Strict, scoped A2 campaign overrides. Release defaults never change."""
from contextlib import contextmanager
from contextvars import ContextVar
import hashlib
import json

from draftkv.backbones.contracts import BackboneError

_ACTIVE = ContextVar('draftkv_a2_experiment', default=None)
CAMPAIGN = 'draftkv_a2_20261108_014305'
LINGBOT_ABLATION_CAMPAIGN = 'draftkv_lingbot_ablation_s6_20261108'
SOLAR_ABLATION_CAMPAIGN = 'draftkv_solar_ablation_l15_20261108'


def camera_only_policy(backbone):
    """Corrected camera-only allocation, independent of Full/native windows."""
    from pathlib import Path
    path = Path(__file__).resolve().parents[4] / 'configs/algorithms/draftkv/a2_frozen_defaults.json'
    models = json.loads(path.read_text())['camera_only']['models']
    if backbone not in models:
        raise BackboneError('CAMERA_POSES_MISSING', 'Camera-only requires native camera geometry')
    policy = models[backbone]
    return dict(backbone=backbone, retrieval_ablation='camera-only',
                budget={key: policy[key] for key in ('sink', 'local', 'retrieved', 'current')},
                protect_recent_frames=policy['protect_recent_frames'],
                camera_top_chunks=policy['camera_top_chunks'])


def override_sha(record):
    return hashlib.sha256(json.dumps(record, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def validate_override(settings, backbone):
    record = settings.get('a2_experiment_override')
    if settings.get('retrieval_ablation') == 'camera-only':
        if record is not None or 'a2_experiment_override_sha256' in settings:
            raise BackboneError('BLOCKED_CONFIGURATION',
                                'obsolete erroneous Camera-only override; remove it and use corrected release defaults')
        if any(settings.get(key, False) for key in
               ('early_fix', 'early_memory_fix', 'temporal_floor', 'candidate_backfill')):
            raise BackboneError('BLOCKED_CONFIGURATION', 'Camera-only forbids fallback/backfill')
        return camera_only_policy(backbone)
    if record is None:
        if 'a2_experiment_override_sha256' in settings:
            raise BackboneError('BLOCKED_CONFIGURATION', 'override hash without override')
        return None
    if isinstance(record, dict) and record.get('schema_version') == 2:
        return _validate_lingbot_ablation_override(settings, backbone, record)
    if isinstance(record, dict) and record.get('schema_version') == 3:
        return _validate_solar_ablation_override(settings, backbone, record)
    fields = {'schema_version', 'campaign_id', 'backbone', 'budget',
              'protect_recent_frames', 'camera_top_chunks'}
    if (not isinstance(record, dict) or set(record) != fields
            or type(record['schema_version']) is not int or record['schema_version'] != 1
            or record['campaign_id'] != CAMPAIGN or record['backbone'] != backbone
            or backbone not in ('LingBot1.3B', 'LingBot14B', 'SolarWM33B')
            or settings.get('retrieval_ablation', 'full') != 'full'
            or settings.get('a2_experiment_override_sha256') != override_sha(record)):
        raise BackboneError('BLOCKED_CONFIGURATION', 'unbound/invalid A2 campaign override')
    b = record['budget']; solar = backbone == 'SolarWM33B'
    if (not isinstance(b, dict) or set(b) != {'sink', 'local', 'retrieved', 'current'}
            or any(type(v) is not int or v < 0 for v in b.values())
            or b['local'] != 0 or b['current'] != (5 if solar else 4)
            or sum(b.values()) != (30 if solar else 18)
            or b['sink'] not in ((0, 5, 10) if solar else (1, 3, 5, 6))
            or type(record['protect_recent_frames']) is not int
            or record['protect_recent_frames'] not in ((3, 5, 7) if solar else (2, 3, 4))
            or record['protect_recent_frames'] > b['retrieved']
            or type(record['camera_top_chunks']) is not int
            or record['camera_top_chunks'] not in ((5, 7, 9) if solar else (2, 3, 4, 5))):
        raise BackboneError('KV_CONSTRAINT', 'unsupported A2 budget/P/K combination')
    forbidden = ('gpu_prepack', 'gpu_device_route', 'early_fix', 'early_memory_fix',
                 'temporal_floor', 'candidate_backfill')
    if any(settings.get(k, False) for k in forbidden):
        raise BackboneError('BLOCKED_CONFIGURATION', 'override cannot enable alternate algorithm paths')
    for key in ('camera_top_chunks', 'protect_recent_frames'):
        if key in settings and settings[key] != record[key]:
            raise BackboneError('BLOCKED_CONFIGURATION', 'conflicting effective A2 parameter')
    return json.loads(json.dumps(record))


def _validate_lingbot_ablation_override(settings, backbone, record):
    """Retain the historical Draft-only allocation; Camera-only is corrected."""
    fields = {'schema_version', 'campaign_id', 'backbone', 'retrieval_ablation',
              'budget', 'protect_recent_frames', 'camera_top_chunks'}
    mode = settings.get('retrieval_ablation')
    if (set(record) != fields or type(record.get('schema_version')) is not int
            or record.get('schema_version') != 2
            or record.get('campaign_id') != LINGBOT_ABLATION_CAMPAIGN
            or backbone not in ('LingBot1.3B', 'LingBot14B')
            or record.get('backbone') != backbone
            or mode != 'draft-only'
            or record.get('retrieval_ablation') != mode
            or settings.get('a2_experiment_override_sha256') != override_sha(record)):
        raise BackboneError('BLOCKED_CONFIGURATION', 'unbound LingBot S6 ablation override')
    budget = record['budget']
    retrieved, top = 8, 0
    if (not isinstance(budget, dict)
            or budget != dict(sink=6, local=0, retrieved=retrieved, current=4)
            or any(type(v) is not int for v in budget.values())
            or type(record['protect_recent_frames']) is not int
            or record['protect_recent_frames'] != 3
            or type(record['camera_top_chunks']) is not int
            or record['camera_top_chunks'] != top):
        raise BackboneError('KV_CONSTRAINT', 'unsupported LingBot S6 ablation budget/P/K')
    forbidden = ('gpu_prepack', 'gpu_device_route', 'early_fix', 'early_memory_fix',
                 'temporal_floor', 'candidate_backfill')
    if any(settings.get(key, False) for key in forbidden):
        raise BackboneError('BLOCKED_CONFIGURATION', 'ablation override cannot enable alternate paths')
    for key in ('camera_top_chunks', 'protect_recent_frames'):
        if key in settings and settings[key] != record[key]:
            raise BackboneError('BLOCKED_CONFIGURATION', 'conflicting LingBot ablation parameter')
    return json.loads(json.dumps(record))


def _validate_solar_ablation_override(settings, backbone, record):
    """Retain unchanged Solar Draft-only R25/P5, mode/SHA bound."""
    fields = {'schema_version', 'campaign_id', 'backbone', 'retrieval_ablation',
              'budget', 'protect_recent_frames', 'camera_top_chunks'}
    mode = settings.get('retrieval_ablation')
    if (set(record) != fields or type(record.get('schema_version')) is not int
            or record.get('schema_version') != 3
            or record.get('campaign_id') != SOLAR_ABLATION_CAMPAIGN
            or backbone != 'SolarWM33B' or record.get('backbone') != backbone
            or mode != 'draft-only'
            or record.get('retrieval_ablation') != mode
            or settings.get('a2_experiment_override_sha256') != override_sha(record)):
        raise BackboneError('BLOCKED_CONFIGURATION', 'unbound Solar ablation override')
    local, retrieved, p, k = 0, 25, 5, 0
    budget = record['budget']
    if (not isinstance(budget, dict)
            or budget != dict(sink=0, local=local, retrieved=retrieved, current=5)
            or any(type(v) is not int for v in budget.values())
            or type(record['protect_recent_frames']) is not int
            or record['protect_recent_frames'] != p
            or type(record['camera_top_chunks']) is not int
            or record['camera_top_chunks'] != k):
        raise BackboneError('KV_CONSTRAINT', 'unsupported Solar ablation budget/P/K')
    forbidden = ('gpu_prepack', 'gpu_device_route', 'early_fix', 'early_memory_fix',
                 'temporal_floor', 'candidate_backfill')
    if any(settings.get(key, False) for key in forbidden):
        raise BackboneError('BLOCKED_CONFIGURATION', 'Solar ablation cannot enable alternate paths')
    for key in ('camera_top_chunks', 'protect_recent_frames'):
        if key in settings and settings[key] != record[key]:
            raise BackboneError('BLOCKED_CONFIGURATION', 'conflicting Solar ablation parameter')
    return json.loads(json.dumps(record))


def active_override(backbone):
    record = _ACTIVE.get()
    return record if record is not None and record['backbone'] == backbone else None


@contextmanager
def experiment_scope(settings, backbone):
    record = validate_override(settings or {}, backbone)
    token = _ACTIVE.set(record)
    try:
        yield record
    finally:
        _ACTIVE.reset(token)


def require_active(settings, backbone):
    record = validate_override(settings, backbone)
    if record != active_override(backbone):
        raise BackboneError('BLOCKED_CONFIGURATION', 'A2 override scope absent or inconsistent')
    return record


def experimental_14b_hook(host):
    """Change admission only; inherit the frozen BFO, selector and consumer."""
    from pathlib import Path
    import numpy as np
    import torch
    from .camera_frustum_v2 import CameraGatedBlockDraftHook
    from .block_backbone import BlockDraftKVHook
    from .camera_frustum_multibackbone import CameraGatedMultiBackboneHook

    class Experimental14BHook(CameraGatedBlockDraftHook):
        # Same frozen BFO and tie breaks, with generic Top-K admission.
        _camera_scope = CameraGatedMultiBackboneHook._camera_scope
        def __init__(self, core):
            policy = require_active(core.plan['algorithm_plugin']['config'], core.spec.backbone)
            if policy is None or core.spec.backbone != 'LingBot14B' or core.spec.chunk != 4:
                raise BackboneError('BLOCKED_CONFIGURATION', 'experimental 14B hook needs scoped C4 policy')
            self.camera_top_chunks = policy['camera_top_chunks']
            self.camera_poses = None
            self._camera_chunk_cache = {}
            self._camera_events = set()
            camera_dir = Path(core.plan['request']['camera_dir'])
            intrinsics = np.load(camera_dir / 'intrinsics.npy', allow_pickle=False)
            if intrinsics.shape != (1, 4):
                raise BackboneError('CAMERA_INTRINSICS_SHAPE', 'WorldKV-Frustum-v2 expects [fx,fy,cx,cy]')
            self._frustum_intrinsics = torch.from_numpy(intrinsics[0].copy()).float()
            self._frustum_width = int(core.plan['request']['width'])
            self._frustum_height = int(core.plan['request']['height'])
            BlockDraftKVHook.__init__(self, core)

    return Experimental14BHook(host)


def archive_page_policy(settings, backbone):
    """Safe default for all DraftKV archives; old scoped records stay valid."""
    record = settings.get('campaign_archive_pages')
    digest = settings.get('campaign_archive_pages_sha256')
    if record is None:
        if digest is not None:
            raise BackboneError('BLOCKED_CONFIGURATION', 'archive policy hash without record')
        return 'base4k'
    if (record != dict(campaign_id=CAMPAIGN, backbone=backbone, policy='base4k')
            or backbone not in ('LingBot1.3B', 'LingBot14B', 'SolarWM33B')
            or digest != override_sha(record)):
        raise BackboneError('BLOCKED_CONFIGURATION', 'unbound campaign archive page policy')
    return 'base4k'
