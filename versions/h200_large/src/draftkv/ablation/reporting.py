"""Bounded trace-to-manifest accounting; never infer unmeasured GPU durations."""
import json
from pathlib import Path


def record_route(host, config, context, selection):
    history = selection.metadata.get('history_budget')
    if history is None:
        pending = selection.metadata.get('draftkv_plan', {})
        history = pending.get('position', {}).get('history_budget')
    host.trace('ablation_actual_context',ablation_profile=config['profile'],layer=context.layer,
               start_frame=context.start_frame,current_frames=context.current_frames,
               frame_tokens=context.frame_tokens,active_kv_tokens=selection.keys.shape[1],
               selected_history_frames=history,history_residency=config['history']['residency'],
               selector=config['retrieval']['selector'],granularity=config['retrieval']['granularity'],
               position_mode=config['retrieval']['position_mode'])


def finalize_manifest(folder, result):
    """Single streaming pass; keep counts/extrema, not a second trace in RAM."""
    count = 0
    low = high = None
    final_counters = None
    for path in sorted((Path(folder)/'traces').glob('backbone_rank*.jsonl')):
        with path.open() as f:
            for line in f:
                row = json.loads(line)
                event = row.get('event')
                if event == 'ablation_actual_context':
                    tokens = row['active_kv_tokens']
                    low = tokens if low is None else min(low,tokens)
                    high = tokens if high is None else max(high,tokens)
                    count += 1
                elif event == 'ablation_prefetch_summary':
                    final_counters = row
    result['actual_context_budget_summary'] = dict(observed_layers=count,min_active_kv_tokens=low,max_active_kv_tokens=high)
    if final_counters:
        for key in result['metrics']:
            if key in final_counters:
                result['metrics'][key] = final_counters[key]
    # CUDA duration fields remain null unless a measured instrument explicitly
    # supplied them. Counters are actual transfer accounting, not timings.
    return result
