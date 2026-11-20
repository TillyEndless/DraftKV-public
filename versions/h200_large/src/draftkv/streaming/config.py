"""CPU contracts; no retrieval, RoPE, SP reshaping or model policy lives here."""
from __future__ import annotations
import hashlib
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ARCHIVE = ROOT / 'vendor/b01-original.tar.gz'
FAMILIES = {'H100': '9.0', 'H200': '9.0', 'B200': '10.0', 'RTX4090': '8.9', 'RTX5090': '12.0'}


def consumer_for(algorithm):
    if algorithm not in ('native', 'draftkv', 'shadowkv', 'infinigen', 'infllm', 'freekv', 'worldkv', 'deep-forcing', 'future-forcing'):
        raise ValueError(f'Unknown algorithm: {algorithm}')
    return 'StreamingFA2' if algorithm == 'draftkv' else 'NativeFA2'


def verify_sources():
    manifest = json.loads((ROOT / 'source_manifest.json').read_text())
    expected = manifest['sha256']
    if hashlib.sha256(ARCHIVE.read_bytes()).hexdigest() != manifest['archive_sha256']:
        raise ValueError('B01 original archive digest mismatch')
    with tarfile.open(ARCHIVE, 'r:gz') as archive:
        actual = {}
        for member in archive:
            if not member.isfile() or Path(member.name).is_absolute() or '..' in Path(member.name).parts or member.name in actual:
                raise ValueError('Unsafe B01 archive member')
            actual[member.name] = hashlib.sha256(archive.extractfile(member).read()).hexdigest()
    if actual != expected:
        raise ValueError('B01 source tree differs from original source manifest')
    digest = hashlib.sha256(json.dumps(expected, sort_keys=True).encode()).hexdigest()
    if digest != manifest['snapshot_sha256']:
        raise ValueError('B01 snapshot digest mismatch')
    return manifest


def frame_frontiers(block_n, sink_tokens, frame_tokens, frames):
    if block_n not in (32, 64, 128) or min(sink_tokens, frame_tokens, frames) <= 0:
        raise ValueError('Unsupported tile size or empty history')
    first = sink_tokens // block_n
    last = (sink_tokens + frame_tokens * frames - 1) // block_n
    updates = [(rank, first if rank == 0 else (sink_tokens + rank * frame_tokens + block_n - 1) // block_n)
               for rank in reversed(range(frames))]
    return first, last, updates


def transfer_schedule(frame_ids, *, heads, head_group, sink_tokens, frame_tokens, block_n):
    """IDs retain caller order; rank denotes packed position, never absolute frame ID."""
    ids = tuple(frame_ids)
    if not ids or any(type(i) is not int or i < 0 for i in ids) or len(set(ids)) != len(ids):
        raise ValueError('Unique nonnegative Frame IDs required')
    if type(heads) is not int or type(head_group) is not int or min(heads, head_group) < 1 or heads % head_group:
        raise ValueError('Head group must divide actual local heads')
    first, last, updates = frame_frontiers(block_n, sink_tokens, frame_tokens, len(ids))
    return {'first': first, 'last': last, 'initial': last + 1,
            'transfers': [{'group': g, 'rank': rank, 'frame_id': ids[rank], 'frontier': value}
                          for g in range(heads // head_group) for rank, value in updates]}


def canonical_runtime_source_sha256():
    """Identity of the B01 source archive actually built by bench1_build_b01.py."""
    from draftkv.vendor.b01.integrity import ROOT as VENDOR_ROOT, digest as vendor_digest, verify_sources as verify_vendor_sources
    verify_vendor_sources()
    return vendor_digest(VENDOR_ROOT / "source_snapshot.tar.gz")

def select_config(family, *, autotune=None, local_heads=None):
    """Fixed source configs are never tuned on deployment. No fabricated winners."""
    if family not in FAMILIES:
        raise ValueError('UnsupportedArchitecture: ' + family)
    policy = json.loads((ROOT / 'hardware_profiles.json').read_text())[family]
    if local_heads is not None:
        if type(local_heads) is not int or local_heads < 1:
            raise ValueError('actual positive local heads required')
        for recipe in policy.get('shape_recipes', ()):
            if local_heads in recipe['local_heads']:
                return {**policy, **recipe}
        allowed = policy.get('capability', {}).get('local_heads')
        if allowed is not None and local_heads not in allowed:
            raise ValueError('No frozen schedule for this provider/local-head count; exact target acceptance required')
    if family == 'H100':
        # Same original SM90 Token Streaming FA2 producer/consumer used in
        # the H200 B-compact route. No implicit switch to old Autotune,
        # ready, or frontier_opt consumers, even when a stale cache exists.
        # H100 hardware acceptance still requires an actual device run.
        parameters = policy.get('parameters')
        validate_parameters(parameters)
        if (parameters != json.loads((ROOT / 'hardware_profiles.json').read_text())['H200']['parameters']
                or parameters['consumer'] != 'original'):
            raise ValueError('H100/H200 canonical SM90 Streaming FA2 config drift')
        if autotune is not None:
            raise ValueError('H100 production does not load legacy Autotune consumer records')
        return {**policy, 'parameters': dict(parameters),
                'status': 'source_bound_sm90_token_streaming_gpu_unverified'}
    if family == 'B200':
        if autotune is None:
            return {**policy, 'parameters': None, 'status': 'pending_hardware_acceptance'}
        required = ('device_family', 'parameters', 'source_snapshot_sha256', 'evidence_sha256', 'evidence_path', 'accepted')
        if any(k not in autotune for k in required) or autotune['device_family'] != family or autotune['accepted'] is not True:
            raise ValueError('Autotune result must be accepted on the actual target family')
        if autotune['source_snapshot_sha256'] != canonical_runtime_source_sha256():
            raise ValueError('Autotune source snapshot differs from canonical release B01 source')
        evidence = Path(autotune['evidence_path'])
        if not evidence.is_file() or hashlib.sha256(evidence.read_bytes()).hexdigest() != autotune['evidence_sha256']:
            raise ValueError('Autotune evidence hash mismatch')
        validate_parameters(autotune['parameters'])
        return {**policy, **autotune, 'status': 'accepted_external_result'}
    return policy


def source_bytes(relative):
    manifest = json.loads((ROOT / 'source_manifest.json').read_text())
    if relative not in manifest['sha256']: raise ValueError('Unknown original source path')
    with tarfile.open(ARCHIVE, 'r:gz') as archive:
        data = archive.extractfile(relative).read()
    if hashlib.sha256(data).hexdigest() != manifest['sha256'][relative]:
        raise ValueError('Original source hash mismatch: ' + relative)
    return data


def extract_sources(destination):
    verify_sources()
    destination = Path(destination)
    with tarfile.open(ARCHIVE, 'r:gz') as archive:
        for member in archive:
            # verify_sources already rejected non-files, traversal and links.
            target = destination / member.name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.extractfile(member).read())
    return destination


def validate_parameters(parameters):
    """Reject settings without an exact implementation in the original consumer."""
    consumer=parameters.get('consumer')
    if consumer not in ('original','ready','frontier_opt','autotune'):
        raise ValueError('UnsupportedArchitecture: original consumer implementation unavailable')
    m=parameters.get('block_m',64); n=parameters.get('block_n'); splits=parameters.get('splits',1)
    if consumer=='autotune':
        if (type(m) is not int or type(n) is not int or type(splits) is not int
                or m not in (64,128) or n not in (32,64,128) or not 1<=splits<=16
                or (splits>1 and m!=64)):
            raise ValueError('UnsupportedArchitecture: tile/split not compiled by original Autotune provider')
    elif (n!=(32 if consumer=='ready' else 64) or m!=64 or splits!=1
          or (consumer=='frontier_opt' and parameters.get('variant')!=9)):
        raise ValueError('UnsupportedArchitecture: parameters do not match the compiled kernel tile/variant')
    group=parameters.get('head_group');gate=parameters.get('gate_tiles')
    if (type(group) is not int or not 0<group<2**31
            or (consumer!='autotune' and group not in (1,2,3,4,6,12))
            or type(gate) is not int or not 0<gate<2**31 or gate&(gate-1)):
        raise ValueError('UnsupportedArchitecture: unsupported group/gate parameters')
    if (parameters.get('host_pages') not in ('pytorch','base4k','thp2m','thp2m_slab')
            or parameters.get('dispatch') not in ('all','early') or parameters.get('layout','grouped')!='grouped'):
        raise ValueError('UnsupportedArchitecture: page/dispatch/split configuration unavailable in this bridge')
    return parameters


def host_pages_policy(parameters, policy='systems'):
    """Select original pinned storage without changing the frozen system recipe.

    Quality changes only host backing. Systems never falls back after a failed
    THP allocation. Both routes retain the canonical native producer/consumer.
    """
    validate_parameters(parameters)
    if policy not in ('quality', 'systems'):
        raise ValueError('host_pages_policy must be quality or systems')
    result = dict(parameters)
    if policy == 'quality':
        result['host_pages'] = 'pytorch'
    return result


def host_pages_provenance(parameters, policy='systems'):
    if policy not in ('quality', 'systems'):
        raise ValueError('host_pages_policy must be quality or systems')
    if policy == 'quality' and parameters['host_pages'] != 'pytorch':
        raise ValueError('quality host storage must use original PyTorch pinned memory')
    return {'host_pages_backend': parameters['host_pages'],
            'host_pages_policy': policy,
            'host_pages_role': 'correctness_only' if policy == 'quality' else 'performance_profile'}
