"""Select a preserved execution context; delegate to its unchanged CLI."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    runtimes = json.loads((ROOT / 'runtimes.json').read_text())
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument('operation', choices=['list', 'show', 'plan', 'preflight', 'run', 'evaluate', 'status', 'static-check', 'dependencies'])
    parser.add_argument('--backbone', choices=list(runtimes))
    parser.add_argument('--profile', choices=['off', 'summary'], default='off')
    if not argv or argv == ['--help']:
        print('Usage: reproduce.sh {list,plan,preflight,run,evaluate,dependencies} --backbone BACKBONE [original options]\n'
              'Backbones: ' + ', '.join(runtimes) + '\nProfile: off (default), summary\n'
              'Native workers, profiling, selection and attention are preserved source bytes.')
        return 0
    args, _ = parser.parse_known_args(argv)
    if args.operation != 'list' and args.backbone is None:
        parser.error('--backbone is required')
    backbone = args.backbone or 'LongLive1.3B'
    source = ROOT / 'verified' / backbone / 'public' / 'src'
    env = dict(os.environ)
    # The original selected CLI owns model planning, launch and measurement.
    env['PYTHONPATH'] = str(source) + (os.pathsep + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    return subprocess.call([sys.executable, '-m', 'draftkv.cli', *argv], env=env)


if __name__ == '__main__':
    raise SystemExit(main())
