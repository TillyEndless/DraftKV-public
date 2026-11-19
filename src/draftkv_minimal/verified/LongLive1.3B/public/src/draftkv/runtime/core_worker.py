"""Execute the public Full core with an explicitly supplied pinned native worker.

This adapter is invoked only by an explicit run operation, after preflight.
It preserves the source-locked original native model and decoder. The public
core overlay is separately fingerprinted; a source digest is never a GPU claim.
"""
from importlib import import_module
from pathlib import Path
import argparse
import os
import sys

from draftkv.algorithms.draftkv.config import config
from draftkv.algorithms.draftkv.core import aliases_for, verify_core
from draftkv.runtime.io import read, require, write
from draftkv.streaming.dependency import verify_b01
from draftkv.streaming.integrity import verify_provider


def activate(backbone, source_root):
    """No filesystem search: only a verified, caller-supplied external root."""
    c = config(backbone)
    verify_provider(c, source_root)
    evidence = verify_core(backbone)
    root = Path(source_root).resolve()
    os.environ['DRAFTKV_EXTERNAL_RUNTIME_ROOT'] = str(root)
    # Parent namespaces may contain both public control code and explicitly
    # authorized model glue. Every selected core module is overlaid explicitly.
    for name in ('draftkv', 'draftkv.backbones', 'draftkv.runtime', 'draftkv.algorithms', 'draftkv.algorithms.draftkv', 'draftkv.streaming'):
        package = import_module(name)
        path = str(root / 'src' / Path(name.replace('.', '/')))
        if path not in package.__path__:
            package.__path__.append(path)
    aliases = aliases_for(backbone)
    modules = {canonical: import_module(public) for canonical, public in aliases.items()}
    for suffix in ('retrieval', 'hardware'):
        name = 'draftkv.retrieval' if suffix == 'retrieval' else 'draftkv.hardware_profiles'
        sys.modules[name] = import_module('draftkv.algorithms.draftkv.historical.' + evidence['profile'] + '.' + suffix)
    for canonical, module in modules.items():
        sys.modules[canonical] = module
        parent, _, child = canonical.rpartition('.')
        if parent in sys.modules:setattr(sys.modules[parent], child, module)
    return {**evidence, 'overlay': aliases, 'native_provider': 'EXACT_EXTERNAL_SOURCE_BYTES_VERIFIED'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release-plan', required=True)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--decode-latents')
    args = parser.parse_args(argv)
    release = read(args.release_plan)
    assets = read(release['asset_config'])
    verify_b01(assets)
    require(release['mode'] == 'run', 'Evaluation cannot invoke generation')
    current = verify_core(release['backbone'])
    require(release['public_core'] == current, 'Public core binding changed after planning')
    worker = read(args.plan)
    require(worker['algorithm'] == 'draftkv' and worker['backbone'] == release['backbone'], 'Worker algorithm/backbone drift')
    require(worker['sp'] == release['runtime_sp'], 'Worker SP drift')
    from draftkv.runtime import profiling
    profiling.configure(release.get('profile_mode','off'))
    os.environ['DRAFTKV_PUBLIC_RELEASE_PLAN'] = str(Path(args.release_plan).resolve())
    evidence = activate(release['backbone'], assets['source_root'])
    write(Path(release['workspace']) / ('public-core-rank' + os.environ.get('RANK', '0') + '.json'), evidence)
    if args.decode_latents:
        require(release['backbone']=='SolarWM33B','Independent H3 decode requires SolarWM')
        sys.argv=['public-h3-decode','--plan',str(Path(args.plan).resolve()),'--latents',str(Path(args.decode_latents).resolve())]
        import_module('draftkv.algorithms.draftkv.historical.orbia.backbones.solarwm_decode').main()
        return
    sys.argv = ['draftkv.backbones.worker', '--plan', str(Path(args.plan).resolve())]
    from draftkv.runtime.native_measurement import install
    native_worker = import_module('draftkv.backbones.worker')
    install(native_worker, worker, release, assets['source_root'])
    native_worker._cli_main(['--plan', str(Path(args.plan).resolve())])


if __name__ == '__main__':
    main()
