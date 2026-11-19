"""Public Full implementations, isolated by their verified historical version."""
from importlib import import_module, resources
from pathlib import Path
import ast
import hashlib
import json

from draftkv.runtime.io import require

PREFIX = 'draftkv.algorithms.draftkv.historical'


def profile_for(backbone):
    require(backbone in ('LongLive1.3B', 'LingBot1.3B', 'LingBot14B', 'SolarWM33B'), 'Unknown backbone')
    return 'longlive' if backbone == 'LongLive1.3B' else 'orbia'


def manifest():
    return json.loads(resources.files('draftkv').joinpath('data/core_sources.json').read_text())


def verify_core(backbone):
    profile = profile_for(backbone)
    records = [r for r in manifest()['files'] if r['profile'] in (profile, 'shared')]
    require(bool(records), 'Missing public algorithm source manifest')
    checked = []
    for record in records:
        rel = Path(record['public_destination']).relative_to('src/draftkv')
        raw = resources.files('draftkv').joinpath(str(rel)).read_bytes()
        require(hashlib.sha256(raw).hexdigest() == record['public_sha256'], 'PUBLIC_CORE_HASH_MISMATCH: ' + str(rel))
        if str(rel).endswith(".py"):ast.parse(raw)
        checked.append(record['public_destination'])
    # Check implementation presence, not merely a budget/interface record.
    required = {'algorithm/backbone.py': 'DraftKVHook', 'algorithm/block_backbone.py': 'BlockDraftKVHook',
                'retrieval/draftattention2_importance.py': 'spatial8x16_mean_max',
                'retrieval/block_topk_p3.py': 'select_blocks', 'streaming/pipeline.py': 'run_layer'}
    if profile == 'orbia':
        required['algorithm/camera_frustum_v2.py'] = 'bilateral_frustum_scores'
    root = resources.files(PREFIX + '.' + profile)
    for rel, symbol in required.items():
        tree = ast.parse(root.joinpath(rel).read_text())
        node = next((n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == symbol), None)
        require(node is not None and len(node.body) > 1, 'Missing actual core implementation: ' + symbol)
    digest = hashlib.sha256(json.dumps([(r['public_destination'], r['public_sha256']) for r in records], separators=(',', ':')).encode()).hexdigest()
    return {'status': 'DRAFTKV_CORE_SOURCE_INCLUDED', 'profile': profile,
            'historical_revision': manifest()['historical_revisions'][profile],
            'source_files_verified': len(checked), 'manifest_sha256': digest,
            'runtime_equivalence_scope': 'CPU fixtures and normalized selected source bodies; GPU execution untested',
            'gpu_tested': False}


def entry_for(backbone):
    verify_core(backbone)
    return import_module(PREFIX + '.' + profile_for(backbone) + '.algorithm.entry')


def aliases_for(backbone):
    """Explicit canonical import overlay used by the authorized native worker."""
    profile = profile_for(backbone)
    base = PREFIX + '.' + profile
    result = {}
    for row in manifest()['files']:
        if row['profile'] != profile:
            continue
        if not row['public_destination'].endswith('.py') or row['public_destination'].endswith('/__init__.py'):continue
        suffix = row['public_destination'].split('/historical/' + profile + '/', 1)[1][:-3].replace('/', '.')
        for public, canonical in [('algorithm.', 'draftkv.algorithms.draftkv.'), ('retrieval.', 'draftkv.retrieval.'),
                                  ('streaming.', 'draftkv.streaming.'), ('backbones.', 'draftkv.backbones.'),
                                  ('hardware.', 'draftkv.hardware_profiles.')]:
            if suffix.startswith(public):
                result[canonical + suffix[len(public):]] = base + '.' + suffix
                break
    result['draftkv.algorithms.draftkv.entry'] = base + '.algorithm.entry'
    return result
