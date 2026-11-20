"""CPU-only budget, history and memory planning. Estimates are not measurements."""
from dataclasses import asdict
from .config import AblationError, deltas, digest, resolve_attention_backend
from .instrumentation import METRICS


def budgets(config, spec, history_frames):
    from draftkv.backbones.budgets import effective_budget
    from draftkv.algorithms.draftkv.block_policy import applied_protection
    b = effective_budget(spec, 'draftkv')
    sink = min(history_frames, b.sink)
    eligible = history_frames-sink
    history = min(b.retrieved, eligible)
    protected = applied_protection(spec.backbone, history) if history else 0
    selector = config['retrieval']['selector']
    if selector == 'all_history':
        history = eligible if config['history']['max_frames'] is None else min(eligible, config['history']['max_frames'])
        local, remote = history, 0
    elif selector == 'local_only':
        history = history if config['history']['local_window_mode'] == 'capacity_fair' else protected
        local, remote = history, 0
    else:
        local, remote = protected, history-protected
    return dict(production=b.to_dict(), sink=sink, local=local, retrieved=remote, current=b.current,
                active_frames=sink+history+b.current, selected_history_frames=history,
                historical_candidates=eligible)


def memory_estimate(config, spec, *, history_frames, frame_tokens):
    for name, value, lower in (('history_frames', history_frames, 0), ('frame_tokens', frame_tokens, 1)):
        if type(value) is not int or value < lower:
            raise AblationError(f'{name} must be integer >= {lower}')
    b = budgets(config, spec, history_frames)
    # BF16 K + BF16 V; includes all layers, not just one attention allocation.
    retained = history_frames
    if config['retrieval']['selector']=='all_history' and config['history']['max_frames'] is not None:
        retained = b['sink']+b['selected_history_frames']
    kv = retained*frame_tokens*spec.heads*spec.head_dim*2*2*spec.layers
    raw = retained*frame_tokens*spec.heads*(64 if spec.family == 'solarwm' else 44)*2*spec.layers
    gpu = kv if config['history']['residency'] == 'gpu' else 0
    host = kv if config['history']['residency'] == 'host' else 0
    if config['retrieval']['position_mode'] == 'chrono_compact':
        if config['history']['residency'] == 'gpu':
            gpu += raw
        else:
            host += raw
    fullkeys = kv//2 if config['history']['searchable_index'] == 'full_keys' and config['history']['fullkey_residency'] == 'gpu' else 0
    gpu += fullkeys
    native_sidecar = raw if config['retrieval']['selector']=='exact_region_relevance' and config['retrieval']['position_mode']=='chrono_compact' and config['history']['fullkey_residency']=='host_fetch' else 0
    host += native_sidecar
    selected = b['selected_history_frames']*frame_tokens*spec.heads*spec.head_dim*4
    active = b['active_frames']*frame_tokens*spec.heads*spec.head_dim*4
    # At most one prediction per layer; retired transfers retain buffers only
    # until events complete. Native attention workspace/model/activations unknown.
    oracle_temporary = history_frames*frame_tokens*spec.heads*spec.head_dim*4 if config['retrieval']['selector']=='exact_region_relevance' else 0
    cache = selected*spec.layers if config['attention']['prefetch_schedule'] not in ('production', 'serial') else 0
    return dict(history_kv_bytes=kv, gpu_persistent_history_bytes=gpu, host_persistent_history_bytes=host,
                fullkey_search_gpu_bytes=fullkeys, prepared_oracle_native_temporal_host_upper_bytes=native_sidecar,
                selected_history_bytes_per_layer=selected,
                active_kv_bytes_per_layer=active, speculative_receive_upper_bytes=cache,
                oracle_full_key_temporary_bytes_per_layer=oracle_temporary,
                estimated_gpu_kv_peak_bytes=gpu+active+(0 if config['retrieval']['selector']=='all_history' else selected)+cache+oracle_temporary,
                excludes=['weights', 'model_activations', 'FA_workspace', 'allocator_fragmentation',
                          'compact_descriptor_bank', 'oracle_QK_tile_workspace', 'CPU_packing_and_registration_rounding'],
                estimate_only=True, precision='BF16', frame_tokens=frame_tokens, history_frames=history_frames,
                retained_history_frames=retained)


def check_memory(config, spec, *, history_frames, frame_tokens, device_memory_gib=None):
    estimate = memory_estimate(config, spec, history_frames=history_frames, frame_tokens=frame_tokens)
    lower = estimate['estimated_gpu_kv_peak_bytes']
    limit = config['safety']['max_history_gib']*1024**3
    if device_memory_gib is not None:
        if not 0 < device_memory_gib < float('inf'):
            raise AblationError('device memory must be positive')
        limit = min(limit, device_memory_gib*1024**3)
    if lower > limit and not config['safety']['allow_over_limit']:
        raise AblationError(f'KV memory safety rejection: estimated GPU KV working set {lower} bytes > ceiling {int(limit)}; '
                            'weights/activations/workspace are additional. Explicit safety.allow_over_limit required.')
    return estimate


def manifest(config, spec, *, benchmark, device, history_frames, frame_tokens, git_sha,
             config_sha256, production_settings=None, device_memory_gib=None):
    memory = check_memory(config, spec, history_frames=history_frames, frame_tokens=frame_tokens,
                          device_memory_gib=device_memory_gib)
    backend = resolve_attention_backend(device)
    if production_settings is not None:
        grouped = production_settings.get('gpu_device_route') and production_settings.get('headgroup_pipeline')
        fullkv = config['retrieval']['selector'] == 'all_history'
        native_canonical = backend in ('FA3','native_FA2')
        if native_canonical and production_settings.get('headgroup_pipeline') is False and not fullkv:
            raise AblationError('native architecture backend disabled by headgroup_pipeline=false')
        if grouped or fullkv or native_canonical:
            backend = resolve_attention_backend(device, stable_backend='native_FA2')
    return dict(schema='draftkv.mechanism_ablation.v1', is_ablation=True, ablation_profile=config['profile'],
                production_semantics_changed=deltas(config), git_sha=git_sha, config_sha256=config_sha256,
                resolved_config_sha256=digest(config), backbone=spec.backbone, benchmark=benchmark,
                device=device, precision='BF16', resolved=config, active_budget=budgets(config, spec, history_frames),
                history_memory_estimate=memory, attention_backend=backend,
                production_settings=production_settings,
                execution_provider='canonical_block' if config['profile'] != 'draftkv_full' else 'unchanged_production',
                oracle='A2: exact full-resolution region-level relevance' if config['retrieval']['selector']=='exact_region_relevance' else None,
                schedule_exactness='speculative_with_authoritative_route_correction' if 'speculative' in config['attention']['prefetch_schedule'] else 'production_or_serial',
                prediction_source='previous_denoise_same_layer_route' if 'speculative' in config['attention']['prefetch_schedule'] else None,
                metrics={k: config['attention']['prefetch_schedule'] if k=='prefetch_schedule' else None for k in METRICS},
                status='STATIC_PLANNED_UNMEASURED', measurements_available=False)
