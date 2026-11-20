"""Read-only byte locks for reused providers; CPU checks never import Torch."""
import hashlib
import json
from pathlib import Path
from draftkv.backbones.contracts import BackboneError, external_path

REPO = Path(__file__).resolve().parents[4]


def verify_sources(algorithm, root=None):
    root = Path(root or REPO)
    lock = root / 'configs' / 'algorithms' / algorithm / 'source_lock.json'
    if not lock.is_file():
        raise BackboneError('BLOCKED_SOURCE', f'missing {algorithm} source inventory {lock}')
    data = json.loads(lock.read_text())
    checked = {}
    for relative, digest in data['files'].items():
        path = root / relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise BackboneError('SOURCE_ABI_DRIFT', f'{algorithm} source differs: {relative}')
        checked[relative] = digest
    return checked


def settings(config):
    return config.get('algorithm_plugin', config.get('backbone_config', {}).get('algorithm_plugin', {})).get('config', {})


def admit(config, root, algorithm):
    backbone = config.get('backbone')
    benchmark = config.get('benchmark', 'bench1')
    supported = {'bench1': ('LongLive1.3B',),
                 'bench2': ('LingBot1.3B', 'LingBot14B', 'SolarWM33B')}
    if backbone not in supported.get(benchmark, ()):
        raise BackboneError('BLOCKED_INPUT_PROTOCOL', f'{algorithm} unsupported benchmark/backbone pair')
    if (config.get('algorithm', algorithm) != algorithm
            or (benchmark == 'bench2' and config.get('algorithm') != algorithm)):
        raise BackboneError('BLOCKED_ALGORITHM_SEMANTICS', 'provider identity differs from group')
    checked = verify_sources(algorithm, root)
    cfg = settings(config)
    if algorithm == 'infinigen' and (cfg.get('partial_ratio', .10) != .10 or cfg.get('alpha', 1) != 1):
        raise BackboneError('BLOCKED_CONFIGURATION', 'release InfiniGen-FrameScore requires partial_ratio=0.10 and alpha=1')
    return {'algorithm': algorithm, 'source_files': checked, 'consumer': 'native_fa2',
            'checkpoint_forward': False, 'video_e2e': False}
