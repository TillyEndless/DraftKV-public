"""Event-ordered copy-engine transport for immutable selected payload pages.

Speculative caches hold already position-processed immutable pages. Identity
includes exact source token order AND compact slot, so a position change is a
miss and gets corrected. No stale/native-vs-Chrono mixing or repeated rotation.
"""
from dataclasses import dataclass


@dataclass
class Ticket:
    source: object
    pair: object
    event: object


class TorchPageTransport:
    def __init__(self, device, instrumentation):
        import torch
        self.device = device
        self.ranges = instrumentation
        self.stream = torch.cuda.Stream(device=device) if device.type == 'cuda' else None
        self.retired = []

    def submit(self, page, label, *, speculative=False):
        import torch
        from contextlib import nullcontext
        context = torch.cuda.stream(self.stream) if self.stream is not None else nullcontext()
        with context, self.ranges.range(label, stream=self.stream):
            pair = tuple(x.to(self.device, non_blocking=self.stream is not None, copy=True) for x in page.payload)
            event = None
            if self.stream is not None:
                event = torch.cuda.Event()
                event.record(self.stream)
        self.reap()
        return Ticket(page.payload, pair, event)

    def wait(self, ticket, *, synchronize=False):
        import torch
        with self.ranges.range('wait_for_kv', stream=torch.cuda.current_stream(self.device) if self.stream is not None else None):
            if ticket.event is not None:
                if synchronize:
                    ticket.event.synchronize()
                else:
                    torch.cuda.current_stream(self.device).wait_event(ticket.event)
                # The pair is then read on the consumer stream. Keep allocator
                # ownership through all Q CTAs, not just the DMA stream.
                for x in ticket.pair:
                    x.record_stream(torch.cuda.current_stream(self.device))

    def release(self, ticket, *, wasted=False):
        # Keep both pinned sources and allocations until DMA completion; GPU
        # record_stream additionally covers asynchronous consumer lifetime.
        from contextlib import nullcontext
        with self.ranges.range('prefetch_wasted') if wasted else nullcontext():
            self.retired.append(ticket)
            self.reap()

    def reap(self):
        self.retired = [t for t in self.retired if t.event is not None and not t.event.query()]

    def close(self):
        if self.stream is not None:
            self.stream.synchronize()
        self.retired.clear()
