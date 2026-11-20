"""Exact H3 rolling cache on CPU, materialized only for the current layer."""


class CpuRawKVCache:
    def __init__(self, layers, device):
        self.layers = layers
        self.device = device

    def _entries(self, layer, chunk_index):
        entries = self.layers.get(layer, [])
        if [e[0] for e in entries] != list(range(max(0, chunk_index-5), chunk_index)):
            raise RuntimeError('H3 CPU KV cache frame order differs')
        return entries

    def history(self, layer, chunk_index):
        return [(i, k.to(self.device, non_blocking=False), v.to(self.device, non_blocking=False))
                for i, k, v in self._entries(layer, chunk_index)]

    def commit(self, layer, chunk_index, key, value):
        import torch
        if torch.is_grad_enabled():
            raise RuntimeError('H3 raw KV commits must be detached no-grad forwards')
        entries = self._entries(layer, chunk_index)
        self.layers[layer] = [*entries, (chunk_index,
            key.detach().to('cpu', copy=True), value.detach().to('cpu', copy=True))][-5:]

    def clear(self):
        self.layers.clear()


def release_provider_registries(core):
    """Unregister only this finished model's caches, never another owner.

    Provider telemetry registries hold strong references to GPU retrieval banks.
    Dropping the model alone leaves those banks alive during independent decode.
    """
    hook = getattr(getattr(core, 'session', None), 'hook', None)
    provider = getattr(hook, 'provider', None)
    attr = getattr(hook, 'cache_attr', None)
    if provider is None:
        return
    if getattr(hook, 'active', None) is not None or getattr(hook, 'pending', {}):
        raise RuntimeError('cannot release provider during a model transaction')
    # WorldKV creates a private module namespace whose compression callback
    # closes over this owner. Unregister that namespace so it cannot retain
    # the model's attention modules after the sampler is dropped.
    import sys
    name = getattr(provider, '__name__', '')
    if name.startswith('_draftkv_worldkv_') and sys.modules.get(name) is provider:
        prefix = name.rsplit('.', 1)[0]
        for key in tuple(sys.modules):
            if key == prefix or key.startswith(prefix + '.'):
                del sys.modules[key]
    if attr is None:
        return
    owned = {id(getattr(m, attr)) for m in hook.modules if hasattr(m, attr)}
    for name in ('SHADOW_CACHES', 'SPECULATIVE_CACHES'):
        registry = getattr(provider, name, None)
        if registry is not None:
            registry[:] = [cache for cache in registry if id(cache) not in owned]
