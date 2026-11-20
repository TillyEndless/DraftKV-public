"""Background preparation of immutable, registered BF16 clean-KV arenas.

Allocation begins inside the first real route. Only the next bounded arena is
prepared ahead of consumption; every published slice lives with its pool.
"""
import mmap
import time
from concurrent.futures import ThreadPoolExecutor


class Arena:
    def __init__(self, device, layers, chunks, frames, ft, *, heads=12, frame_counts=None,
                 width=256, page_policy='base4k'):
        import torch
        begin = time.perf_counter_ns()
        self.page_policy = page_policy
        torch.cuda.set_device(device)
        self.layers, self.chunks = layers, chunks
        self.frame_counts = tuple(frame_counts or (frames,) * chunks)
        if len(self.frame_counts) != chunks or any(not 0 < f <= frames for f in self.frame_counts):
            raise ValueError('invalid clean frame capacities')
        per_frame = ft * heads * width
        self.elements = frames * per_frame
        self.offsets = [0]
        for count in self.frame_counts:
            self.offsets.append(self.offsets[-1] + count * per_frame)
        count = layers * self.offsets[-1]
        self.bytes = count * 2
        self.mapping = mmap.mmap(-1, self.bytes,
            flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS, prot=mmap.PROT_READ | mmap.PROT_WRITE)
        from draftkv.algorithms.draftkv.registered_pages import advise_archive
        advise_archive(self.mapping, page_policy)
        self.storage = torch.frombuffer(self.mapping, dtype=torch.bfloat16, count=count)
        self.storage.zero_()
        touched = time.perf_counter_ns()
        status = torch.cuda.cudart().cudaHostRegister(self.storage.data_ptr(), self.bytes, 0)
        if status != torch.cuda.cudart().cudaError.success:
            raise RuntimeError(f'arena registration failed: {status}')
        if not self.storage.is_pinned():
            raise RuntimeError('arena is not pinned')
        self.metrics = dict(bytes=self.bytes, touch_ms=(touched-begin)/1e6,
            register_ms=(time.perf_counter_ns()-touched)/1e6,
            total_ms=(time.perf_counter_ns()-begin)/1e6)

    def slice(self, layer, chunk, elements):
        if not 0 <= layer < self.layers or not 0 <= chunk < self.chunks:
            raise ValueError('arena layer/chunk outside allocation')
        capacity = self.offsets[chunk+1] - self.offsets[chunk]
        if not 0 < elements <= capacity:
            raise ValueError('arena chunk capacity exceeded')
        begin = layer * self.offsets[-1] + self.offsets[chunk]
        return self.storage[begin:begin+elements]


class ArenaPool:
    def __init__(self, device, layers, frames, ft, chunks=4, max_generations=None, *,
                 heads=12, max_frames=None, width=256, page_policy='base4k'):
        values = (layers, frames, ft, chunks, heads, width)
        if any(type(v) is not int or v <= 0 for v in values):
            raise ValueError('arena geometry must contain positive integers')
        if max_frames is not None:
            if type(max_frames) is not int or max_frames <= 0:
                raise ValueError('invalid archive frame limit')
            generations = (max_frames + frames - 1) // frames
            if max_generations is not None and max_generations != generations:
                raise ValueError('archive frame and generation limits disagree')
            max_generations = generations
        if max_generations is not None and (type(max_generations) is not int or max_generations <= 0):
            raise ValueError('invalid clean generation limit')
        self.device, self.layers, self.frames, self.ft = device, layers, frames, ft
        self.chunks, self.heads = chunks, heads
        if page_policy not in ('base4k', 'thp2m'):
            raise ValueError('invalid archive page policy')
        self.width, self.page_policy = width, page_policy
        self.max_generations, self.max_frames = max_generations, max_frames
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='draftkv-arena')
        self.futures, self.reported = {}, set()
        self.frame_capacities = {}
        self.closed = False
        self.request(0)

    def request(self, index):
        if type(index) is not int or index < 0:
            raise ValueError('invalid arena index')
        if self.closed:
            raise RuntimeError('archive pool is closed')
        first = index * self.chunks
        if self.max_generations is not None and first >= self.max_generations:
            return
        if index in self.futures:
            return
        chunks = self.chunks
        if self.max_generations is not None:
            chunks = min(chunks, self.max_generations-first)
        counts = tuple(self.frames if self.max_frames is None else
            min(self.frames, self.max_frames-(first+i)*self.frames) for i in range(chunks))
        self.frame_capacities[index] = counts
        self.futures[index] = self.executor.submit(Arena, self.device, self.layers,
            chunks, self.frames, self.ft, heads=self.heads, frame_counts=counts,
            width=self.width, page_policy=self.page_policy)

    def slice(self, layer, generation, elements):
        if type(generation) is not int or generation < 0:
            raise ValueError('invalid clean generation')
        if self.max_generations is not None and generation >= self.max_generations:
            raise ValueError('clean generation exceeds declared frame request')
        index, chunk = divmod(generation, self.chunks)
        self.request(index)
        begin = time.perf_counter_ns()
        arena = self.futures[index].result()
        wait = (time.perf_counter_ns()-begin)/1e6
        view = arena.slice(layer, chunk, elements)
        if chunk >= self.chunks//2:
            self.request(index+1)
        return view, arena, wait, index

    @property
    def bytes(self):
        """Ready/registered storage only; do not wait for background preparation."""
        return sum(f.result().bytes for f in self.futures.values()
                   if f.done() and not f.cancelled() and f.exception() is None)

    @property
    def requested_bytes(self):
        return sum(sum(v) for v in self.frame_capacities.values()) * self.layers * self.ft * self.heads * self.width * 2

    def _metric(self, name):
        return sum(f.result().metrics.get(name, 0) for f in self.futures.values()
                   if f.done() and not f.cancelled() and f.exception() is None)

    @property
    def touch_ms(self):
        return self._metric('touch_ms')

    @property
    def register_ms(self):
        return self._metric('register_ms')

    def close(self, wait=True):
        self.closed = True
        self.executor.shutdown(wait=wait, cancel_futures=True)


def prepacked_archive_slice(hook, context):
    """Same immutable pool for older GPU block/group prepared providers."""
    import atexit
    if hook.active.get('recache', False):
        return None  # never overwrite already-published sink slices
    pool = getattr(hook, '_prepacked_archive_pool', None)
    if pool is None:
        capacity = hook.host.plan['history_plan']['capacity_frames']
        pool = ArenaPool(context.key.device,hook.host.spec.layers,hook.host.spec.chunk,
                         context.frame_tokens,heads=context.key.shape[2],max_frames=capacity,
                         page_policy=hook.archive_page_policy)
        hook._prepacked_archive_pool = pool
        hook._clean_registered_owners.append(pool)
        atexit.register(pool.close)
    if context.start_frame % pool.frames:
        raise ValueError('prepacked archive chunk alignment mismatch')
    slab, owner, wait_ms, index = pool.slice(context.layer,
        context.start_frame//pool.frames,context.key.numel()*2)
    hook.host.trace('draftkv_prepacked_archive_slice',layer=context.layer,
        start_frame=context.start_frame,arena_index=index,archive_wait_ms=wait_ms,
        page_policy=pool.page_policy,allocation='bounded_ahead_of_use_arena')
    return slab, owner
