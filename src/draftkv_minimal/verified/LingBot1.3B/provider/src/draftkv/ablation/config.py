"""Strict ablation resolution; disabled files return the original production object.

JSON documents are used in .yaml files (a YAML 1.2 subset), keeping planning
stdlib-only. General YAML is accepted when PyYAML is installed, with duplicate
mapping keys rejected in both formats.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path


class AblationError(ValueError):
    pass


BASE = dict(retrieval=dict(selector='draft', granularity='block', position_mode='chrono_compact'),
            history=dict(residency='host', searchable_index='draft_bank', local_window_mode='production',
                         max_frames=None, fullkey_residency='host_fetch'),
            attention=dict(consumer='production', prefetch_schedule='production'),
            safety=dict(max_history_gib=128, allow_over_limit=False))
PROFILES = {
    'draftkv_full': {},
    'wo_draft_retrieval_local': {'retrieval.selector': 'local_only'},
    'wo_draft_retrieval_capacity_fair': {'retrieval.selector': 'local_only', 'history.local_window_mode': 'capacity_fair'},
    'fullkv': {'retrieval.selector': 'all_history', 'retrieval.position_mode': 'native_post_rope',
               'history.residency': 'gpu', 'history.searchable_index': 'none'},
    'exact_relevance_selector': {'retrieval.selector': 'exact_region_relevance'},
    'frame_topk_equal_budget': {'retrieval.granularity': 'frame'},
    'gpu_resident_history': {'history.residency': 'gpu'},
    'wo_draft_bank_fullkey_search': {'retrieval.selector': 'exact_region_relevance', 'history.searchable_index': 'full_keys'},
    'draft_bank_no_remote_retrieval': {'retrieval.selector': 'local_only'},
    'stream_serial': {'attention.prefetch_schedule': 'serial'},
    'prefetch_same_step_next_layer': {'attention.prefetch_schedule': 'same_step_next_layer'},
    'prefetch_same_step_next_layer_speculative': {'attention.prefetch_schedule': 'same_step_next_layer_speculative'},
    'prefetch_cross_step_same_layer_speculative': {'attention.prefetch_schedule': 'cross_step_same_layer_speculative'},
    'prefetch_hybrid': {'attention.prefetch_schedule': 'hybrid_speculative'},
}
MODES = {'retrieval.selector': {'draft', 'local_only', 'all_history', 'exact_region_relevance'},
         'retrieval.granularity': {'block', 'frame'},
         'retrieval.position_mode': {'chrono_compact', 'native_post_rope'},
         'history.residency': {'host', 'gpu'},
         'history.searchable_index': {'draft_bank', 'full_keys', 'none'},
         'history.fullkey_residency': {'host_fetch', 'gpu'},
         'history.local_window_mode': {'production', 'capacity_fair'},
         'attention.consumer': {'production'},
         'attention.prefetch_schedule': {'production', 'serial', 'same_step_next_layer',
             'same_step_next_layer_speculative', 'cross_step_same_layer_speculative', 'hybrid_speculative'}}


def flatten(data):
    return {f'{section}.{key}': value for section, fields in data.items() for key, value in fields.items()}


def _unique(pairs):
    result = {}
    for k, v in pairs:
        if k in result:
            raise AblationError(f'duplicate key: {k}')
        result[k] = v
    return result


def read_config(path):
    raw = Path(path).read_text()
    try:
        return json.loads(raw, object_pairs_hook=_unique)
    except json.JSONDecodeError:
        try:
            import yaml
        except ImportError as e:
            raise AblationError('General YAML requires PyYAML; bundled profiles use JSON-compatible YAML') from e
        class Loader(yaml.SafeLoader):
            pass
        def mapping(loader, node):
            return _unique([(loader.construct_object(k), loader.construct_object(v)) for k, v in node.value])
        Loader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, mapping)
        return yaml.load(raw, Loader=Loader)


def resolve(document, production=None, *, requested_profile=None):
    """Inactive stored fields are ignored; a conflicting explicit request fails."""
    if not isinstance(document, dict) or set(document) != {'ablation'}:
        raise AblationError('expected only top-level ablation mapping')
    data = document['ablation']
    if not isinstance(data, dict) or type(data.get('enabled')) is not bool:
        raise AblationError('ablation.enabled must be explicit boolean')
    if not data['enabled']:
        if requested_profile not in (None, 'production'):
            raise AblationError('disabled ablation conflicts with explicitly requested profile')
        return production
    if set(data) - {'enabled', 'profile', *BASE}:
        raise AblationError(f'unknown ablation fields: {set(data)-{"enabled", "profile", *BASE}}')
    profile = data.get('profile')
    if requested_profile is not None and requested_profile != profile:
        raise AblationError('explicit profile conflicts with config')
    if profile is not None and (not isinstance(profile,str) or profile not in PROFILES):
        raise AblationError(f'unknown profile: {profile}')
    if profile is None and any(s not in data or not isinstance(data[s],dict) or set(data[s]) != set(BASE[s]) for s in BASE):
        raise AblationError('enabled ablation needs a known profile or complete fields')
    result = deepcopy(BASE)
    for path, value in PROFILES.get(profile, {}).items():
        section, key = path.split('.')
        result[section][key] = value
    for section in BASE:
        if section in data:
            fields = data[section]
            if not isinstance(fields, dict) or set(fields) - set(BASE[section]):
                raise AblationError(f'unknown fields in {section}')
            result[section].update(fields)
    for path, legal in MODES.items():
        section, key = path.split('.')
        if not isinstance(result[section][key],str) or result[section][key] not in legal:
            raise AblationError(f'unknown mode {path}: {result[section][key]}')
    r, h, a, s = (result[x] for x in ('retrieval', 'history', 'attention', 'safety'))
    if a['prefetch_schedule'] == 'same_step_next_layer':
        raise AblationError('Exact next-layer route is unavailable before future query; use same_step_next_layer_speculative')
    if r['selector'] == 'all_history' and (h['searchable_index'] != 'none' or h['residency'] != 'gpu'
                                          or r['position_mode'] != 'native_post_rope'):
        raise AblationError('FullKV requires all history, no top-K/index, GPU residency and native positions')
    if h['searchable_index'] == 'none' and r['selector'] != 'all_history':
        raise AblationError('no searchable index is legal only for FullKV')
    if h['searchable_index'] == 'full_keys' and r['selector'] != 'exact_region_relevance':
        raise AblationError('full-key search requires exact_region_relevance')
    if h['fullkey_residency'] == 'gpu' and h['searchable_index'] != 'full_keys':
        raise AblationError('fullkey residency applies only to full-key search')
    if h['local_window_mode'] == 'capacity_fair' and r['selector'] != 'local_only':
        raise AblationError('capacity-fair window requires local-only selector')
    if h['max_frames'] is not None and (type(h['max_frames']) is not int or h['max_frames'] < 1 or r['selector'] != 'all_history'):
        raise AblationError('positive max_frames is supported only for explicit FullKV horizon')
    if type(s['allow_over_limit']) is not bool or type(s['max_history_gib']) not in (int, float) or not 0 < s['max_history_gib'] < float('inf'):
        raise AblationError('invalid history safety ceiling/override')
    if profile:
        # Named causal profiles cannot silently become a different experiment.
        for path, value in flatten(result).items():
            if path.startswith('safety.') or path in ('history.max_frames', 'history.fullkey_residency'):
                continue
            if value != PROFILES[profile].get(path, flatten(BASE)[path]):
                raise AblationError(f'profile {profile} conflicts with {path}')
    if r['selector'] == 'exact_region_relevance':
        from .oracle import ExactRegionRelevanceOracle
        ExactRegionRelevanceOracle()  # availability check only; no torch/model import
    result.update(enabled=True, profile=profile or 'custom')
    return result


def deltas(config):
    return {k: v for k, v in flatten({s: config[s] for s in BASE}).items() if v != flatten(BASE)[k]}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def resolve_attention_backend(device, config=None, *, stable_backend=None):
    """Device names/capabilities are supplied, never probed by static planning."""
    key = str(device).lower().replace(' ', '').replace('_', '')
    if any(x in key for x in ('h100', 'h200', 'hopper')) or device in ((9, 0), 'sm90'):
        return 'FA3'
    if '4090' in key or device in ((8, 9), 'sm89'):
        return stable_backend or 'production_b01'
    if '5090' in key or device in ((12, 0), 'sm120'):
        return stable_backend or 'native_FA2'
    raise AblationError(f'explicit supported device/capability required: {device}')
