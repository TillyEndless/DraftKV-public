"""Optional runtime ranges. Importing this module never imports torch/CUDA."""
from contextlib import contextmanager

LABELS = ('draft_score', 'route', 'index_select', 'cpu_pack', 'gpu_gather',
          'h2d_prefetch', 'h2d_correction', 'wait_for_kv', 'attention', 'prefetch_wasted', 'rope')
METRICS = ('prefetch_schedule', 'selected_bytes', 'prefetched_bytes', 'useful_prefetched_bytes',
           'wasted_prefetched_bytes', 'correction_bytes', 'route_hit_rate', 'h2d_sum_ms',
           'h2d_exposed_ms', 'attention_ms', 'block_wall_ms')


class Instrumentation:
    def __init__(self, *, nvtx=False):
        self.nvtx = nvtx
        self.events = []

    @contextmanager
    def range(self, label, *, stream=None):
        if label not in LABELS:
            raise ValueError(f'unknown instrumentation label: {label}')
        import torch
        with torch.profiler.record_function('ablation/'+label):
            if self.nvtx:
                torch.cuda.nvtx.range_push('ablation/'+label)
            pair = None
            if stream is not None:
                pair = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                pair[0].record(stream)
            try:
                yield
            finally:
                if pair:
                    pair[1].record(stream)
                    self.events.append((label, *pair))
                if self.nvtx:
                    torch.cuda.nvtx.range_pop()

    def collect_completed(self):
        ready, pending = [], []
        for label, a, b in self.events:
            if b.query():
                ready.append(dict(label=label, cuda_envelope_ms=a.elapsed_time(b)))
            else:
                pending.append((label, a, b))
        self.events = pending
        return ready


def report_envelopes(host, ranges):
    """Drain completed events at existing clean fences; bounded label aggregates.

    These are CUDA queue/wait envelopes, not pure DMA/SM activity and not E2E.
    """
    totals = {}
    for row in ranges.collect_completed():
        item = totals.setdefault(row['label'],dict(interval_count=0,cuda_envelope_sum_ms=0.0))
        item['interval_count'] += 1
        item['cuda_envelope_sum_ms'] += row['cuda_envelope_ms']
    if totals:
        host.trace('ablation_cuda_envelopes',labels=totals,
                   interpretation='sums_of_cuda_envelopes_may_overlap_include_waits_not_active_time')
