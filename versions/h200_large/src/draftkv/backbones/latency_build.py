"""Separate extension construction from measured runtime initialization.

Build in a disposable process, never warm the sampler's modules/kernels. A
process-wide observer also covers compiler launches from background threads.
"""
import hashlib
import importlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import threading
import time

CACHE_VARIABLES = ('DRAFTKV_PREPACK_BUILD', 'DRAFTKV_PREPACK_SM120_BUILD',
                   'DRAFTKV_ROUTE_SM120_BUILD')
POLICY = 'prebuilt_extensions_runtime_cold_included_v1'
_OBSERVER = None
_AUDIT_INSTALLED = False
_LOCK = threading.Lock()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_backends(plan):
    if plan.get('algorithm') != 'draftkv':
        return []
    settings = (plan.get('algorithm_plugin') or {}).get('config', {})
    entry = str((plan.get('algorithm_plugin') or {}).get('entry', ''))
    if 'entry_sm120_route' in entry:
        return ['routed_sm120']
    if 'entry_sm120' in entry:
        return ['prepacked_sm120']
    names = []
    if (settings.get('table2_native_runtime') or settings.get('headgroup_pipeline')
            or settings.get('gpu_device_route')):
        names.append('device_route')
    if settings.get('gpu_prepack'):
        names.append('prepacked')
    return names


def prepare_profile_builds(plan, trace=None):
    """Called after environment sanitization and before model construction."""
    if not plan.get('gpu_profile', False):
        return None
    if '_latency_build_preparation' in plan:
        return plan['_latency_build_preparation']
    for name in CACHE_VARIABLES:
        if name in os.environ and not Path(os.environ[name]).is_absolute():
            raise ValueError(f'{name} must bind an absolute build directory')
    folder = Path(plan['run_dir']) / 'traces'
    folder.mkdir(parents=True, exist_ok=True)
    rank = int(os.environ.get('RANK', '0'))
    target = folder / f'latency_build_prepare_rank{rank}.json'
    backends = build_backends(plan)
    bindings = {name: os.environ.get(name) for name in CACHE_VARIABLES}
    receipt = dict(schema_version=1, status='READY', policy=POLICY,
                   phase='environment_preparation', excluded_from_e2e=True,
                   cache_bindings=bindings, backends=backends, extensions=[])
    if backends:
        job = dict(backends=backends, output=str(target), local_rank=int(os.environ.get('LOCAL_RANK', '0')),
                   receipt=receipt)
        begin = time.perf_counter()
        proc = subprocess.run([sys.executable, '-m', __name__, '--prepare'],
                              input=json.dumps(job), text=True, capture_output=True)
        log = folder / f'latency_build_prepare_rank{rank}.log'
        log.write_text(proc.stdout + proc.stderr, encoding='utf-8')
        if proc.returncode:
            raise RuntimeError(f'extension environment preparation failed; see {log}')
        receipt = json.loads(target.read_text())
        receipt['preparation_wall_s'] = time.perf_counter() - begin
        if (receipt.get('status') != 'READY' or receipt.get('cache_bindings') != bindings
                or receipt.get('backends') != backends):
            raise RuntimeError('extension preparation receipt differs from sanitized worker')
    else:
        receipt['preparation_wall_s'] = 0.0
    target.write_text(json.dumps(receipt, indent=2) + '\n')
    plan['_latency_build_preparation'] = receipt
    if trace is not None:
        trace('latency_build_prepared', **receipt)
    return receipt


def _prepare_child(job):
    import torch
    torch.cuda.set_device(job['local_rank'])
    capability = torch.cuda.get_device_capability()
    receipt = job['receipt']
    for backend in job['backends']:
        if backend not in ('device_route', 'prepacked', 'prepacked_sm120', 'routed_sm120'):
            raise ValueError('unknown native build backend')
        actual = ('prepacked_sm120' if capability == (12, 0) and backend == 'prepacked'
                  else backend)
        module = importlib.import_module('draftkv.streaming.' + actual).native()
        path = Path(module.__file__).resolve()
        receipt['extensions'].append(dict(name=module.__name__, module_file=str(path),
            sha256=digest(path), source_identity=module.source_identity))
    receipt.update(torch=str(torch.__version__), cuda=torch.version.cuda, capability=list(capability))
    Path(job['output']).write_text(json.dumps(receipt, indent=2) + '\n')


def _audit(event, args):
    observer = _OBSERVER
    if observer is not None and event == 'subprocess.Popen':
        observer.command(args[0], args[1])
    elif observer is not None and event == 'os.system':
        observer.command('/bin/sh', ['sh', '-c', os.fsdecode(args[0])])


class CompilationWatch:
    """Observe builds without changing attention, streams or ready waits."""
    def __init__(self, preparation=None, cpp=None):
        self.preparation = preparation or {}
        self.cpp = cpp
        self.events = []
        self.reasons = []
        self.loaded = []
        self.local = threading.local()
        self.originals = {}

    def reject(self, reason, **fields):
        if reason not in self.reasons:
            self.reasons.append(reason)
        if len(self.events) < 64:
            self.events.append(dict(reason=reason, **fields))

    def command(self, executable, argv):
        args = [os.fsdecode(a) for a in argv] if isinstance(argv, (tuple, list)) else [str(argv)]
        name = Path(os.fsdecode(executable)).name
        queries = {'--version', '-V', '-v', '-dumpversion', '-dumpfullversion'}
        compilers = {'c++', 'g++', 'gcc', 'cc', 'clang', 'clang++', 'nvcc', 'ptxas', 'nvlink', 'ld', 'ld.lld'}
        if name in {'sh', 'bash', 'dash'} and '-c' in args:
            try:
                words = shlex.split(args[args.index('-c')+1])
            except (ValueError, IndexError):
                self.reject('UNPARSEABLE_BUILD_SHELL')
                return
            for i, word in enumerate(words):
                if Path(word).name in compilers | {'ninja'}:
                    self.command(word, words[i:])
        if name in compilers and not (set(args[1:]) and set(args[1:]) <= queries):
            self.reject('COMPILER_DURING_GENERATION', executable=name,
                        arguments=args[1:16])
        if name == 'ninja' and not getattr(self.local, 'ninja', False):
            if not any(a in args for a in ('--version', '-n', '--dry-run', '-t')):
                self.reject('UNVERIFIED_BUILD_DRIVER', executable=name)

    @staticmethod
    def _log(root):
        path = Path(root) / '.ninja_log'
        return path.read_bytes() if path.exists() else b''

    def ninja(self, *args, **kwargs):
        root = kwargs.get('build_directory', args[0] if args else None)
        before = self._log(root)
        begin = time.perf_counter()
        self.local.ninja = True
        try:
            return self.originals['_run_ninja_build'](*args, **kwargs)
        finally:
            self.local.ninja = False
            if self._log(root) != before:
                self.reject('JIT_EXTENSION_BUILD_DURING_GENERATION', build_directory=str(root),
                            build_check_wall_ms=(time.perf_counter()-begin)*1000)

    def import_module(self, *args, **kwargs):
        module = self.originals['_import_module_from_library'](*args, **kwargs)
        path = Path(module if isinstance(module, str) else module.__file__).resolve()
        item = dict(module_file=str(path), sha256=digest(path))
        expected = {r['module_file']: r['sha256'] for r in self.preparation.get('extensions', [])}
        if str(path) not in expected:
            self.reject('UNPREPARED_JIT_EXTENSION', **item)
        elif item['sha256'] != expected[str(path)]:
            self.reject('PREBUILT_BINARY_CHANGED', **item)
        self.loaded.append(item)
        return module

    def __enter__(self):
        global _OBSERVER, _AUDIT_INSTALLED
        with _LOCK:
            if _OBSERVER is not None:
                raise RuntimeError('nested generation compilation observation is unsupported')
            if not _AUDIT_INSTALLED:
                sys.addaudithook(_audit)
                _AUDIT_INSTALLED = True
            _OBSERVER = self
        expected_bindings = self.preparation.get('cache_bindings')
        if expected_bindings is not None and expected_bindings != {
                name: os.environ.get(name) for name in CACHE_VARIABLES}:
            self.reject('CACHE_BINDING_CHANGED_AFTER_PREPARATION')
        try:
            if self.cpp is not None:
                self._attach()
        except BaseException:
            self.__exit__()
            raise
        return self

    def _attach(self):
        for name, wrapper in [('_run_ninja_build', self.ninja),
                              ('_import_module_from_library', self.import_module)]:
            self.originals[name] = getattr(self.cpp, name)
            setattr(self.cpp, name, wrapper)

    def attach_runtime_loader(self, torch):
        # Keep even the first Python build-helper import inside the measured
        # interval. Only groups using a local JIT extension need this helper.
        if self.cpp is None and self.preparation.get('backends') and hasattr(torch, '__path__'):
            self.cpp = importlib.import_module('torch.utils.cpp_extension')
            self._attach()

    def __exit__(self, *error):
        global _OBSERVER
        for name, original in self.originals.items():
            setattr(self.cpp, name, original)
        with _LOCK:
            _OBSERVER = None

    def report(self):
        return dict(policy=POLICY, status='COMPILE_CONTAMINATED' if self.reasons else 'PASS',
                    runtime_cold_included=True, build_time_subtracted=False,
                    reasons=self.reasons, events=self.events, loaded_extensions=self.loaded)


def compilation_watch(torch, preparation=None):
    return CompilationWatch(preparation)


if __name__ == '__main__':
    if sys.argv[1:] != ['--prepare']:
        raise SystemExit('use --prepare with a build job on stdin')
    _prepare_child(json.load(sys.stdin))
