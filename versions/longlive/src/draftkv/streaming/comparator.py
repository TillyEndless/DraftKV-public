"""Explicit serial comparison of the unchanged native B01 execution contract.

The original BoundPlan validates and owns all operands, workspace, poison
quarantine and joins. Only host submission timing changes. No new DMA, Frontier
or attention implementation is introduced, and this is never a profile winner.
"""
from types import SimpleNamespace

from draftkv.vendor.b01.contract import consumer_module, require


def run_serial_reference(plan, q, out, lse, scale):
    """Submit every original C++ DMA before the SAME gated FA2 Consumer.

    For early dispatch the original first/tail enqueue ranges are each called
    once, before Consumer launch. Its later tail callback becomes a checked
    no-op. An event on the actual fetch stream fences all H2D/Frontier writes.
    Original warmup, parameter validation, storage ownership and join survive.
    """
    require(plan.state == 'BOUND', 'serial comparator requires a fresh unpoisoned plan')
    native, modules, spec = plan.native, plan.modules, plan.spec
    completed = plan.torch.cuda.Event()
    plan._retained.append(completed)
    submitted = False
    deferred_tail_seen = False

    class SerialProducer:
        def initialize(self, value):
            return native.initialize(value)

        def enqueue(self, begin, end):
            nonlocal submitted, deferred_tail_seen
            if not submitted:
                require(begin == 0 and end == (1 if spec.dispatch == 'early' else plan.entry_count),
                        'unexpected original first ProducerPlan range')
                native.enqueue(begin, end)
                if spec.dispatch == 'early':
                    native.enqueue(1, plan.entry_count)
                completed.record(plan.streams[0])
                submitted = True
                return
            require(spec.dispatch == 'early' and not deferred_tail_seen
                    and begin == 1 and end == plan.entry_count,
                    'unexpected duplicate ProducerPlan serial submission')
            deferred_tail_seen = True

        def bytes(self):
            return native.bytes()

    name = consumer_module(spec)
    symbol = 'streaming_fwd' if spec.consumer == 'token' else 'fwd'
    consumer = getattr(modules[name], symbol)

    def serial_consumer(*args):
        if args[6] >= args[5]:
            require(submitted, 'serial gated Consumer cannot precede full ProducerPlan submission')
            plan.streams[1].wait_event(completed)
        return consumer(*args)

    plan.native = SerialProducer()
    plan.modules = dict(modules)
    plan.modules[name] = SimpleNamespace(**{symbol: serial_consumer})
    try:
        return plan.run(q, out, lse, scale)
    finally:
        # Native objects and event are retained by the SAME original BoundPlan.
        # It remains solely responsible for join or poison quarantine.
        plan.native, plan.modules = native, modules
