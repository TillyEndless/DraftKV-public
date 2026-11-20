"""Route-authoritative prefetch state machines independent of attention math.

Pages are immutable payloads with semantic identities (source tokens, temporal
slots, clean generation, branch and layer). Correction is mandatory before
consumption. Transfer completion alone is insufficient if position processing
is still outstanding: transport.wait must include that readiness dependency.
"""
from dataclasses import dataclass, field
from .config import AblationError


@dataclass(frozen=True)
class Page:
    identity: object
    nbytes: int
    payload: object = field(default=None, compare=False, repr=False)


@dataclass
class Route:
    layer: int
    pages: tuple

    def __post_init__(self):
        if len({p.identity for p in self.pages}) != len(self.pages) or any(p.nbytes < 1 for p in self.pages):
            raise AblationError('unique positive-sized semantic page identities required')


class PrefetchScheduler:
    """Transport protocol: submit(Page,label)->ticket, wait(ticket), release(ticket).

    wait establishes DMA + position processing visibility before ANY consumer
    load/TMA prefetch. A transport may return stream dependencies without a CPU
    fence; serial explicitly asks for synchronize=True. No busy-wait CTA.
    """
    name = 'serial'
    cross = next_layer = False

    def __init__(self, transport, *, layers):
        if type(layers) is not int or layers < 1:
            raise AblationError('positive layer count required')
        self.transport, self.layers = transport, layers
        self.epoch = None
        self.step = -1
        self.previous = {}
        self.predicted = {}
        self.active = None
        self.ready = None
        self.attention_started = False
        self.counters = dict(selected_bytes=0, prefetched_bytes=0, useful_prefetched_bytes=0,
                             wasted_prefetched_bytes=0, correction_bytes=0)
        self.selected_pages = self.hit_pages = 0

    def on_denoise_step_start(self, step, *, epoch):
        if self.active is not None:
            raise AblationError('cannot advance step while attention is active')
        if epoch != self.epoch:
            self.drain()
            self.previous.clear()
            self.epoch = epoch
        self.step = step

    def on_layer_start(self, layer):
        if self.active is not None or not 0 <= layer < self.layers:
            raise AblationError('invalid layer lifecycle')
        self.active = layer
        self.ready = None
        self.attention_started = False

    def _predict(self, layer):
        if layer not in self.previous or layer in self.predicted:
            return
        route = self.previous[layer]
        self.predicted[layer] = {p.identity: (p, self.transport.submit(p, 'h2d_prefetch', speculative=True)) for p in route.pages}
        self.counters['prefetched_bytes'] += sum(p.nbytes for p in route.pages)

    def on_route_ready(self, route):
        if route.layer != self.active or self.ready is not None:
            raise AblationError('route does not match active layer')
        predicted = self.predicted.pop(route.layer, {})
        actual = {p.identity for p in route.pages}
        for identity, (p, ticket) in predicted.items():
            if identity not in actual:
                self.counters['wasted_prefetched_bytes'] += p.nbytes
                self.transport.release(ticket, wasted=True)
        tickets = []
        for p in route.pages:
            hit = p.identity in predicted
            if hit:
                old, ticket = predicted[p.identity]
                if old.nbytes != p.nbytes:
                    raise AblationError('same semantic page identity changed size')
                self.counters['useful_prefetched_bytes'] += p.nbytes
                self.hit_pages += 1
            else:
                ticket = self.transport.submit(p, 'h2d_correction' if self.cross or self.next_layer else 'h2d_prefetch')
                if self.cross or self.next_layer:
                    self.counters['correction_bytes'] += p.nbytes
            tickets.append((p, ticket))
        self.selected_pages += len(route.pages)
        self.counters['selected_bytes'] += sum(p.nbytes for p in route.pages)
        self.ready = route, tickets
        return tuple(t for _, t in tickets)

    def on_attention_start(self):
        if self.ready is None or self.attention_started:
            raise AblationError('actual route and correction required before unique attention start')
        route, tickets = self.ready
        for _, ticket in tickets:
            self.transport.wait(ticket, synchronize=self.name == 'serial')
        self.attention_started = True
        if self.next_layer and route.layer+1 < self.layers:
            self._predict(route.layer+1)
        # The returned order is EXACTLY actual route order, never prediction order.
        return tuple((p.identity, t) for p, t in tickets)

    def on_attention_end(self):
        if self.ready is None or not self.attention_started:
            raise AblationError('missing attention lifecycle')
        route, tickets = self.ready
        self.previous[route.layer] = route
        for _, ticket in tickets:
            self.transport.release(ticket)
        self.active = self.ready = None
        if self.cross:
            self._predict(route.layer)

    def drain(self):
        """Fence pending copies before dropping pinned sources/receive allocations."""
        for entries in self.predicted.values():
            for p, ticket in entries.values():
                self.counters['wasted_prefetched_bytes'] += p.nbytes
                self.transport.release(ticket, wasted=True)
        self.predicted.clear()

    def metrics(self):
        return dict(prefetch_schedule=self.name, **self.counters,
                    route_hit_rate=self.hit_pages/self.selected_pages if self.selected_pages else None,
                    h2d_sum_ms=None, h2d_exposed_ms=None, attention_ms=None, block_wall_ms=None)


class SerialScheduler(PrefetchScheduler):
    pass


class SameStepNextLayerScheduler(PrefetchScheduler):
    name = 'same_step_next_layer_speculative'
    next_layer = True

    def __init__(self, *args, exact=False, **kwargs):
        if exact:
            raise AblationError('Exact next-layer route is unavailable before future query')
        super().__init__(*args, **kwargs)


class CrossStepSameLayerScheduler(PrefetchScheduler):
    name = 'cross_step_same_layer_speculative'
    cross = True


class HybridScheduler(PrefetchScheduler):
    name = 'hybrid_speculative'
    cross = next_layer = True

    def on_attention_end(self):
        # First-layer prefetch spans denoise steps; subsequent layers are
        # predicted during the preceding layer's current-step attention.
        if self.ready is None or not self.attention_started:
            raise AblationError('missing attention lifecycle')
        route, tickets = self.ready
        self.previous[route.layer] = route
        for _, ticket in tickets:
            self.transport.release(ticket)
        self.active = self.ready = None
        if route.layer == 0:
            self._predict(0)


def make_scheduler(name, transport, *, layers):
    types = {c.name: c for c in (SerialScheduler, SameStepNextLayerScheduler,
                                CrossStepSameLayerScheduler, HybridScheduler)}
    if name not in types:
        raise AblationError(f'unsupported explicit scheduler {name}')
    return types[name](transport, layers=layers)
