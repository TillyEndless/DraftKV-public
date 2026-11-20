"""Canonical B01 binding: one source/cache identity, no ambient extension imports."""
from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
import hashlib
import json
from pathlib import Path

from draftkv.vendor.b01 import PlanSpec, source_generation
from draftkv.vendor.b01.build import compatibility_generation
from draftkv.vendor.b01.integrity import B01Error
from .config import validate_parameters, host_pages_policy, host_pages_provenance


# This original Python contract was bundled with the ext binaries. A change
# only to Python orchestration must not force a recompilation of the unchanged
# C++/CUDA producer and consumer. Every other compatibility file is rehashed.
_ORIGINAL_NATIVE_CONTRACT_SHA256 = '58bbb26af2c008e187d2ad8762864f239023ae09d41070daf7d93a1c7b95485d'
# Python-only opt-in warmed-once bridge: accept only the exact reviewed
# Python contract; the compiled CUDA/C++ source, ABI, and every other
# compatibility file must still match the legacy native build specification.
_PRE_WARMUP_FIX_CONTRACT_SHA256 = '8d8f186734c481d4a932c73fda98042b13c659dabc7b0bafc29b8dddd7b27869'
_WARMUP_FIX_CONTRACT_SHA256 = 'd9c0724d863e73e44d816418d60c429d68ffa7ce653c910feb87e9a9ddc62616'



def _native_build_compatible(spec):
    """Accept a verified legacy binary when only the Python plan bridge changed."""
    from draftkv.vendor.b01.build import compatibility_generation
    from draftkv.vendor.b01.integrity import ROOT, canonical_hash, digest
    expected = spec.get('compatibility_generation')
    if expected == compatibility_generation():
        return True
    files = ('build.py', 'setup_compat.py', 'cache.py', 'contract.py',
             'integrity.py', '_source_lock.py')
    previous = {name: (_ORIGINAL_NATIVE_CONTRACT_SHA256 if name == 'contract.py'
                       else digest(ROOT / name)) for name in files}
    if expected == canonical_hash(previous):
        return True
    if digest(ROOT / 'contract.py') != _WARMUP_FIX_CONTRACT_SHA256:
        return False
    previous['contract.py'] = _PRE_WARMUP_FIX_CONTRACT_SHA256
    return expected == canonical_hash(previous)


def _verified_native_cache(directory, build_spec, abi, *, load_modules=True):
    """Verify the original native binaries without pinning Python-only bridge SHA.

    Reproduces cache.verify_cache receipt, content key, ABI and binary SHA checks,
    but the build's Python contract provenance is validated separately in
    _native_build_compatible() at worker startup.
    """
    from draftkv.vendor.b01.build import MODULES, assert_abi
    from draftkv.vendor.b01.cache import cache_key, _import_modules
    from draftkv.vendor.b01.integrity import confined, digest
    assert_abi(build_spec['abi'], abi)
    directory = Path(directory)
    try:
        record = json.loads((directory / 'receipt.json').read_text())
        key = cache_key(record['spec'], record['toolchain'])
        if (record['schema'] != 1 or record['validation'] != 'BUILD_EXPORTS_ONLY'
                or record['spec'] != build_spec or record['key'] != key
                or directory.name != key):
            raise B01Error('CUDA_ABI', 'original B01 cache receipt identity mismatch')
        if set(record['binaries']) != set(MODULES):
            raise B01Error('CUDA_ABI', 'original native module set mismatch')
        paths = {}
        hashes = {}
        for name, binary in record['binaries'].items():
            if binary['path'] != name + abi['ext_suffix']:
                raise B01Error('CUDA_ABI', 'original native module Python ABI mismatch')
            path = confined(directory, binary['path'])
            checksum = digest(path)
            if not path.stat().st_size or checksum != binary['sha256']:
                raise B01Error('CUDA_ABI', f'original native binary hash mismatch: {name}')
            paths[name] = path
            hashes[name] = checksum
        return paths, hashes, (_import_modules(paths) if load_modules else {})
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise B01Error('CUDA_ABI', f'invalid original binary receipt: {exc}') from exc


def _runtime_abi_for_spec(build_spec, actual):
    """Normalize only an interpreter alias inside the same venv bin directory.

    The canonical B01 build intentionally records the invoked venv Python path
    rather than resolving it, because that path owns site-packages. Deployment
    may invoke the same interpreter through python instead of python3.X. Treat
    those names as one ABI identity only when both paths share the same parent
    directory and resolve to the same existing file. Every other ABI field,
    and every cross-environment Python path, remains strict in verify_cache.
    """
    expected = build_spec.get('abi') if isinstance(build_spec, dict) else None
    if not isinstance(expected, dict):
        return actual
    exp = expected.get('python_executable')
    got = actual.get('python_executable') if isinstance(actual, dict) else None
    if not isinstance(exp, str) or not isinstance(got, str) or exp == got:
        return actual
    ep, gp = Path(exp), Path(got)
    try:
        same_alias = ep.parent == gp.parent and ep.resolve(strict=True) == gp.resolve(strict=True)
    except OSError:
        same_alias = False
    if not same_alias:
        return actual
    return {**actual, 'python_executable': exp}


@dataclass(frozen=True)
class NativeBinding:
    cache_directory: Path
    build_spec: dict
    parameters: dict
    hardware_profile_binding: dict
    root: Path
    alternate_profile_bindings: tuple = ()
    profile_admission: str = 'runtime'
    host_pages_policy: str = 'systems'
    # Shared immutable worker-owned session, retained across dataclasses.replace.
    runtime: object = field(default=None, repr=False, compare=False)

    def verify_build_cache(self):
        """Verify receipt, ABI and binary hashes without importing model Torch.

        read() has already checked the source and supported Python bridge.
        The actual worker still checks its own interpreter ABI and exports.
        """
        return _verified_native_cache(self.cache_directory, self.build_spec,
            self.build_spec['abi'], load_modules=False)[0]

    def prepare_runtime(self):
        """Verify source and binary provenance ONCE before the first Attention.

        Compiled native modules stay imported and bound for this worker. A new
        source/ABI installation requires worker restart and a fresh binding.
        """
        from dataclasses import replace
        import torch
        from draftkv.vendor.b01.cache import check_exports
        from draftkv.vendor.b01.build import current_abi
        from draftkv.vendor.b01.integrity import ROOT as VENDOR_ROOT, digest, source_generation
        from types import SimpleNamespace
        if self.runtime is not None:
            return self
        generation = source_generation()  # Full SHA of archive members, one time.
        if self.build_spec['source_generation'] != generation:
            raise B01Error('CUDA_ABI', 'worker build/source generation mismatch')
        abi = _runtime_abi_for_spec(self.build_spec, current_abi(torch))
        if not _native_build_compatible(self.build_spec):
            raise B01Error('CUDA_ABI', 'native compile inputs differ beyond the Python bridge')
        paths, hashes, modules = _verified_native_cache(self.cache_directory, self.build_spec, abi)
        check_exports(modules)
        from draftkv.vendor.b01.contract import consumer_module
        session = SimpleNamespace(generation=generation, modules=modules,
                                  paths=paths, hashes=hashes, abi=abi,
                                  snapshot_sha=digest(VENDOR_ROOT / 'source_snapshot.tar.gz'))
        return replace(self, runtime=session)

    @property
    def host_pages_metadata(self):
        return host_pages_provenance(self.parameters, self.host_pages_policy)

    @classmethod
    def read(cls, config, root):
        """Read/validate only. No Torch imports, build, GPU probe or downloads."""
        root = Path(root).resolve()
        if not isinstance(config, dict):
            raise B01Error('MissingDependency', 'explicit canonical b01 binding required')
        native = config.get('b01')
        admission = config.get('profile_admission', 'runtime')
        profile = config.get('hardware_profile_binding')
        if not isinstance(native, dict):
            raise B01Error('MissingDependency', 'explicit canonical b01 binding required')
        if admission != 'runtime' and not isinstance(profile, dict):
            raise B01Error('MissingDependency', 'hardware_profile_binding is required only for diagnostic profile admission')
        if profile is None:
            profile = {}
        def path(value):
            if not isinstance(value, str) or not Path(value).is_absolute():
                raise B01Error('PathError', 'B01 cache/spec paths must be explicit absolute paths')
            return Path(value).resolve()
        spec_path = path(native.get('build_spec_path'))
        cache = path(native.get('cache_directory'))
        if not spec_path.is_file() or hashlib.sha256(spec_path.read_bytes()).hexdigest() != native.get('build_spec_sha256'):
            raise B01Error('CUDA_ABI', 'canonical B01 build spec missing or hash mismatch')
        try:
            spec = json.loads(spec_path.read_text())
        except (ValueError, OSError) as exc:
            raise B01Error('CUDA_ABI', 'invalid canonical B01 build spec') from exc
        if spec.get('source_generation') != source_generation() or not _native_build_compatible(spec):
            raise B01Error('CUDA_ABI', 'source or compiled native ABI differs from the verified B01 build')
        parameters = dict(native.get('parameters', {}))
        # SM90's canonical runtime (H100/H200) is the optimized H200
        # original Token-Streaming B01 path, never a stale legacy consumer.
        # The separate offline profiling API can still inspect native
        # alternatives, and 4090/5090/B200 production is unaffected.
        if (spec.get('arch') == '9.0' and admission == 'runtime'
                and parameters.get('consumer') != 'original'):
            raise B01Error('UnsupportedArchitecture',
                           'SM90 production requires H200-compatible original token Streaming FA2; legacy consumer disabled')
        page_policy = config.get('host_pages_policy', 'systems')
        if page_policy == 'quality' and admission != 'runtime':
            raise B01Error('LogicError', 'quality host storage cannot claim a frozen systems profile')
        parameters = host_pages_policy(parameters, page_policy)
        alternates = config.get('hardware_profile_alternates', [])
        if (not isinstance(alternates, list) or len(alternates) > 64 or
                any(not isinstance(value, dict) or not
                    {'catalog', 'request', 'artifact_root'} <= set(value)
                    or set(value) - {'catalog', 'request', 'artifact_root', 'asset_roots'}
                    for value in alternates)):
            raise B01Error('LogicError', 'alternate exact Profile bindings require explicit catalog/request/artifact_root')
        # Alternate paths are not wildcard SM/shape recipes: every candidate
        # still carries a signed Golden and is checked against live QKV/ABI.
        unique = [json.dumps(binding, sort_keys=True) for binding in [profile] + alternates]
        if len(unique) != len(set(unique)):
            raise B01Error('LogicError', 'duplicate exact Profile binding')
        if admission not in ('exact', 'blocker_only', 'runtime'):
            raise B01Error('LogicError', 'unknown profile admission')
        return cls(cache, spec, parameters, dict(profile), root,
                   tuple(dict(value) for value in alternates), admission, page_policy)

    def plan(self, frame_ids, *, frame_tokens, history_start, kv_tokens, layer, layers, heads):
        p = self.parameters
        spec = PlanSpec(tuple(frame_ids), frame_tokens, history_start, kv_tokens,
                        layer, layers, self.build_spec['arch'], self.runtime.generation if self.runtime else source_generation(),
                        self.build_spec['clean_commit'],
                        consumer={'original': 'token', 'ready': 'ready', 'frontier_opt': 'frontier_opt', 'autotune': 'autotune'}[p['consumer']],
                        heads=heads, head_group=p['head_group'], gate_tiles=p['gate_tiles'], variant=p.get('variant', 9),
                        block_m=p.get('block_m',64), tile_n=p['block_n'], splits=p.get('splits',1),
                        dispatch=p['dispatch'], host_pages=p['host_pages'], layout=p.get('layout','grouped'), preverified_source=self.runtime is not None)
        if spec.block_n != p['block_n']:
            raise B01Error('UnsupportedArchitecture', 'profile BlockN differs from the actual original architecture dispatch')
        return spec

    def resident_spec(self, *, frame_tokens, kv_tokens, layer, layers, heads):
        # Empty history is a real ungated consumer mode. Do not invent a frame
        # or ProducerPlan to satisfy the nonempty native Producer contract.
        p = self.parameters
        consumer = {'original': 'token', 'ready': 'ready', 'frontier_opt': 'frontier_opt', 'autotune': 'autotune'}[p['consumer']]
        block_n = p['block_n'] if consumer == 'autotune' else ((32 if self.build_spec['arch'] == '8.9' else 64) if consumer == 'token' else (32 if consumer == 'ready' else 64))
        if block_n != p['block_n']:
            raise B01Error('UnsupportedArchitecture', 'resident consumer BlockN differs from bound profile')
        if consumer in ('ready','frontier_opt') and heads != 12:
            raise B01Error('UnsupportedArchitecture', 'group-major resident consumer requires 12 LOCAL heads')
        validate_parameters(p)
        if type(heads) is not int or heads<1 or (consumer != 'token' and heads%p['head_group']):
            raise B01Error('LogicError', 'head_group must divide actual local heads')
        return SimpleNamespace(frame_ids=(), frame_tokens=frame_tokens, kv_tokens=kv_tokens,
                               layer=layer, layers=layers, heads=heads,
                               consumer=consumer, head_group=p['head_group'], gate_tiles=p['gate_tiles'],
                               block_n=block_n, block_m=p.get('block_m',64), splits=p.get('splits',1),
                               layout=p.get('layout','grouped'), dispatch=p['dispatch'],host_pages=p['host_pages'],
                               variant=p.get('variant', 9), arch=self.build_spec['arch'],
                               generation=self.runtime.generation if self.runtime else source_generation(), groups=1 if consumer == 'token' else heads // p['head_group'])

    @staticmethod
    def _compatible_shape_request(profile, spec, actual_request=None):
        """Blocker-only family check; no per-shape Golden acceptance is claimed.

        Reuse ONLY identical physical GPU/driver, source/kernel SHA, ABI, SP,
        native head geometry and contiguous BF16 Q/K/V layout. Sequence lengths
        are allowed to change; no score/attention or native fallback is swapped.
        """
        from copy import deepcopy
        from draftkv.hardware_profiles.profiles import ProfileError, checked
        reference=profile['key']
        ranks=reference['ranks']
        if (not isinstance(spec.frame_tokens, int) or spec.frame_tokens<=0
                or not isinstance(spec.kv_tokens, int) or spec.kv_tokens<=0
                or spec.kv_tokens%spec.frame_tokens
                or len(ranks)<1 or any(r['shape']['heads']!=spec.heads
                                     or r['shape']['frame_tokens']!=spec.frame_tokens
                                     or r['shape']['kv_heads']!=spec.heads
                                     for r in ranks)):
            raise ProfileError('UNSUPPORTED_SHAPE',
                'blocker-only cannot change BF16 page geometry/head mapping')
        if actual_request is None:
            return
        checked(actual_request,'request')
        expected=deepcopy(reference)
        observed=deepcopy(actual_request)
        if len(observed['ranks'])!=len(expected['ranks']):
            raise ProfileError('BLOCKED_ABI','SP rank count differs in blocker-only mode')
        fields=('q','kv','history_tokens','q_strides','k_strides','v_strides')
        for target,old in zip(observed['ranks'],expected['ranks']):
            o=target['shape']
            if (o['heads']!=spec.heads or o['frame_tokens']!=spec.frame_tokens
                    or o['kv']!=spec.kv_tokens
                    or o['history_tokens']!=len(spec.frame_ids)*spec.frame_tokens
                    or o['q']<=0 or o['q']%spec.frame_tokens
                    or o['kv']<o['q'] or o['kv']<o['history_tokens']
                    or o['layout']!='BSHD' or o['dtype']!='bf16'):
                raise ProfileError('UNSUPPORTED_SHAPE','live QKV geometry differs from native blocker-only PlanSpec')
            h,d=o['heads'],o['dim']
            for side,length in (('q',o['q']),('k',o['kv']),('v',o['kv'])):
                strides=[length*h*d,h*d,d,1]
                if o[side+'_strides']!=strides:
                    raise ProfileError('BLOCKED_ABI','QKV noncontiguous layout forbidden in blocker-only mode')
            for name in fields:
                o[name]=old['shape'][name]
        if observed!=expected:
            raise ProfileError('BLOCKED_ABI',
                'blocker-only permits sequence lengths only, never device/ABI/source/kernel/SP/head changes')

    def verify_profile(self, spec, *, gpu=False, actual_request=None):
        from draftkv.hardware_profiles.deployment import verify_deployment, verify_plan_profile
        from draftkv.hardware_profiles.profiles import ProfileError
        bindings = (self.hardware_profile_binding,) + self.alternate_profile_bindings
        desired = (spec.heads,spec.kv_tokens,spec.frame_tokens,
                   len(spec.frame_ids)*spec.frame_tokens)
        candidates=[]
        for binding in bindings:
            candidate=verify_deployment(binding,self.root,gpu=False)
            shapes=[rank['shape'] for rank in candidate['profile']['key']['ranks']]
            if all((shape['heads'],shape['kv'],shape['frame_tokens'],
                    shape['history_tokens'])==desired for shape in shapes):
                verify_plan_profile(candidate['profile'],spec)
                candidates.append(binding)
        if len(candidates)>1:
            raise ProfileError('UNSUPPORTED_SHAPE','need one unique accepted exact Profile')
        exact=bool(candidates)
        if not exact and self.profile_admission!='blocker_only':
            # Single-profile legacy callers retain clear PlanSpec mismatch
            # diagnostics; absent/ambiguous multi-profile remains fail-closed.
            if len(bindings)==1:
                verify_plan_profile(verify_deployment(bindings[0],self.root)['profile'],spec)
            raise ProfileError('UNSUPPORTED_SHAPE',
                f'need one exact accepted Profile for live PlanSpec {desired}; no shape/SM fallback permitted')
        binding=candidates[0] if exact else bindings[0]
        if exact:
            result=verify_deployment(binding,self.root,gpu=gpu,actual_request=actual_request)
            verify_plan_profile(result['profile'],spec)
            result={**result,'admission_scope':'EXACT_PROFILE_GOLDEN'}
        else:
            result=verify_deployment(binding,self.root,gpu=False)
            verify_plan_profile(result['profile'],spec,require_geometry=False)
            self._compatible_shape_request(result['profile'],spec,actual_request)
            result={**result,'admission_scope':'BLOCKER_ONLY_DYNAMIC_SHAPE_NOT_GOLDEN'}
        config=result['profile']['config']
        if any(config[key]!=self.parameters[key] for key in
               ('head_group','gate_tiles','block_n','host_pages','dispatch')):
            raise B01Error('LogicError','native parameters differ from frozen B01 profile')
        return result

    def live_profile(self, q, k, v, spec, *, sp, validation_only=False):
        """Load actual B01 binary; production runtime has no Profile/Golden gate."""
        import torch
        from draftkv.vendor.b01.build import current_abi
        from draftkv.vendor.b01.cache import verify_cache
        from draftkv.vendor.b01.integrity import ROOT as VENDOR_ROOT, digest
        from draftkv.vendor.b01.contract import consumer_module
        actual_abi = _runtime_abi_for_spec(self.build_spec, current_abi(torch))
        if self.runtime is None:
            paths = verify_cache(self.cache_directory, self.build_spec, actual_abi)
            module = consumer_module(spec)
            kernel_sha = digest(paths[module])
        else:
            # Reuse the one-time full SHA/Receipt/ABI verification. No mutable
            # file/path I/O in the Attention hot path.
            paths = self.runtime.paths
            module = consumer_module(spec)
            kernel_sha = self.runtime.hashes[module]
        if self.profile_admission == 'runtime':
            source_sha = (self.runtime.snapshot_sha if self.runtime else digest(VENDOR_ROOT / 'source_snapshot.tar.gz'))
            live = dict(admission_scope='RUNTIME_NO_PROFILE_GATE',
                        source_snapshot_sha256=source_sha,
                        kernel_module=module, kernel_sha256=kernel_sha,
                        current_abi=current_abi(torch), sp=sp,
                        frame_tokens=spec.frame_tokens,
                        history_tokens=len(spec.frame_ids) * spec.frame_tokens,
                        tensors=[dict(name=name, shape=list(tensor.shape),
                                      stride=list(tensor.stride()),
                                      dtype=str(tensor.dtype), device=str(tensor.device))
                                 for name, tensor in (('q', q), ('k', k), ('v', v))])
            profile_sha = hashlib.sha256(json.dumps(
                live, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
            return {'profile': {'sha256': profile_sha,
                                'config': dict(self.parameters),
                                'kernel': {'sha256': kernel_sha}},
                    'actual_request': live,
                    'admission_scope': 'RUNTIME_NO_PROFILE_GATE'}
        from draftkv.hardware_profiles.profiles import source_digest
        from draftkv.hardware_profiles.deployment import capture_actual_request
        static = self.verify_profile(spec)
        profile = static['profile']
        hashes = {record['sha256'] for record in profile['source_files']}
        if source_generation() not in hashes or digest(VENDOR_ROOT / 'source_snapshot.tar.gz') not in hashes:
            raise B01Error('CUDA_ABI', 'profile must lock canonical B01 source manifest AND source snapshot')
        if kernel_sha != profile['kernel']['sha256']:
            raise B01Error('CUDA_ABI', 'profile kernel differs from the verified consumer actually loaded')
        actual = capture_actual_request(q=q, k=k, v=v, frame_tokens=spec.frame_tokens,
                                        history_tokens=len(spec.frame_ids) * spec.frame_tokens,
                                        source_sha256=source_digest(profile['source_files']), kernel_sha256=kernel_sha,
                                        parallel={'sp': sp, 'strategy': 'local' if sp == 1 else 'ulysses',
                                                  'input_placement': 'attention_local'})
        result = self.verify_profile(spec, gpu=not validation_only, actual_request=actual)
        return {**result, 'actual_request': actual}
