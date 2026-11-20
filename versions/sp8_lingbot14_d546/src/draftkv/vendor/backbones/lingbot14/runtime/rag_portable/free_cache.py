from concurrent.futures import ThreadPoolExecutor
import os
from threading import Lock

import torch

from .freekv_core import correction_mask, page_scores
from .shadow_core import expand_blocks, reuse_indices
from .base import BaseCache, host_copy


_POOL = ThreadPoolExecutor(max_workers=8, thread_name_prefix='portable-freekv')


class PortableFreeKVCache(BaseCache):
    freekv = True

    def __init__(self):
        self.block_size = int(os.environ.get('LINGBOT_FREEKV_PAGE', '4'))
        self.threshold = float(os.environ.get('H3_FREEKV_CORRECTION', '.9'))
        self.reference = os.environ['H3_FREEKV_REFERENCE']
        self.raw, self.banks = {}, {}
        self.pool = _POOL
        self.stats_lock = Lock()
        self.stats = dict(method='FreeKV-LingBot', implementation='quality_adapter_with_streamed_speculation',
                          metadata_residency='cpu_summary_archive_gpu_search',
                          page_size=self.block_size, correction_threshold=self.threshold,
                          page_rep='quest', GQA_policy='avgSM',
                          spatial_correction='any_patch', clean_commits=0, correction_checks=0,
                          corrected_heads=0, speculative_recalls=0, bootstrap_recalls=0,
                          retrieved_forwards=0, fetched_value_tokens=0, reused_tokens=0,
                          forward_history=[])

    @torch.no_grad()
    def promote(self, start, stop, device):
        bank = self.banks[0]
        current = bank.setdefault('promoted_stop', start)
        if stop == current:
            return
        if stop < current or start % self.block_size or stop % self.block_size:
            raise ValueError('Invalid monotonic page-aligned promotion')
        if bank['pending'] is not None:
            raise RuntimeError('Prediction survived a clean chunk boundary')
        bank['last_query'] = None
        chunk_tokens = bank['offloaded'][0].shape[-2]
        parts = []
        for chunk in range(current // chunk_tokens, (stop - 1) // chunk_tokens + 1):
            lower = max(current - chunk * chunk_tokens, 0)
            upper = min(stop - chunk * chunk_tokens, chunk_tokens)
            parts.append(bank['offloaded'][chunk][0, :, lower:upper])
        keys = torch.cat(parts, dim=-2)[None].to(device)
        batch, heads, count, width = keys.shape
        pages = keys.reshape(batch, heads, count // self.block_size, self.block_size, width)
        bank['minimum'].append(host_copy(pages.amin(-2)))
        bank['maximum'].append(host_copy(pages.amax(-2)))
        bank['all_minimum'] = torch.cat(bank['minimum'], dim=-2).to(device)
        bank['all_maximum'] = torch.cat(bank['maximum'], dim=-2).to(device)
        bank['global_start'], bank['promoted_stop'] = start, stop

    def select(self, bank, query, budget):
        scores = page_scores(query, bank['all_minimum'], bank['all_maximum'])
        return expand_blocks(scores.topk(budget // self.block_size, dim=-1).indices, self.block_size) + bank['global_start']

    @torch.no_grad()
    def fetch(self, bank, ids):
        ids = ids[0]
        heads, count = ids.shape
        width = bank['all_minimum'].shape[-1]
        locations, hits = reuse_indices(bank['resident_ids'], ids)
        keys = torch.empty((heads, count, width), device=ids.device, dtype=bank['all_minimum'].dtype)
        values = torch.empty_like(keys)
        if bank['resident_ids'] is not None:
            if ids.is_cuda:
                for field in ('resident_ids', 'resident_k', 'resident_v'):
                    bank[field].record_stream(torch.cuda.current_stream(ids.device))
            for suffix, output in (('k', keys), ('v', values)):
                output[hits] = bank['resident_' + suffix].gather(
                    1, locations.unsqueeze(-1).expand(-1, -1, width))[hits]
        missing_heads, missing_slots = (~hits).nonzero(as_tuple=True)
        missing_ids = ids[missing_heads, missing_slots].cpu()
        host_heads = missing_heads.cpu()
        segment_rows = bank['offloaded'][0].shape[-2]
        capacity = max(ids.numel(), heads * segment_rows * 2)
        host_storage = torch.empty(2 * capacity * width, device='cpu', dtype=keys.dtype,
                                   pin_memory=ids.is_cuda)
        host_buffer = host_storage[:2 * len(missing_ids) * width].view(2, len(missing_ids), width)
        for segment, archive in enumerate(bank['offloaded']):
            mask = missing_ids // segment_rows == segment
            host_buffer[:, mask] = archive[:, host_heads[mask], missing_ids[mask] % segment_rows]
        if len(missing_ids):
            recalled = host_buffer.to(ids.device, non_blocking=True)
            keys[missing_heads, missing_slots] = recalled[0]
            values[missing_heads, missing_slots] = recalled[1]
        bank['resident_ids'], bank['resident_k'], bank['resident_v'] = ids.clone(), keys, values
        with self.stats_lock:
            self.stats['fetched_value_tokens'] += len(missing_ids)
            self.stats['reused_tokens'] += ids.numel() - len(missing_ids)
        return keys.unsqueeze(0), values.unsqueeze(0)

    @torch.no_grad()
    def select_fetch(self, layer, query, budget):
        bank = self.banks[layer]
        if bank['pending'] is None:
            ids = self.select(bank, query, budget)
            keys, values = self.fetch(bank, ids)
            self.stats['bootstrap_recalls'] += 1
        else:
            ids, keys, values, event = bank['pending'].result()
            bank['pending'] = None
            if event is not None:
                torch.cuda.current_stream(query.device).wait_event(event)
                ids.record_stream(torch.cuda.current_stream(query.device))
                keys.record_stream(torch.cuda.current_stream(query.device))
                values.record_stream(torch.cuda.current_stream(query.device))
            correct = correction_mask(query, bank['last_query'], self.threshold, self.reference)
            self.stats['correction_checks'] += correct.numel()
            self.stats['corrected_heads'] += int(correct.sum().item())
            if bool(correct.any()):
                fresh_ids = self.select(bank, query, budget)
                ids = torch.where(correct.unsqueeze(-1), fresh_ids, ids)
                keys, values = self.fetch(bank, ids)
        return ids, keys, values

    def after_attention(self, layer, query, budget, clean):
        bank = self.banks[layer]
        if clean:
            bank['last_query'] = None
            return
        bank['last_query'] = query.detach().clone()
        if not query.is_cuda:
            ids = self.select(bank, bank['last_query'], budget)
            keys, values = self.fetch(bank, ids)
            from concurrent.futures import Future
            future = Future()
            future.set_result((ids, keys, values, None))
            bank['pending'] = future
        else:
            ready = torch.cuda.Event()
            ready.record(torch.cuda.current_stream(query.device))
            if 'stream' not in bank:
                bank['stream'] = torch.cuda.Stream(device=query.device)
            stream = bank['stream']
            forecast_query = bank['last_query']

            @torch.no_grad()
            def forecast():
                with torch.cuda.device(query.device), torch.cuda.stream(stream):
                    stream.wait_event(ready)
                    forecast_query.record_stream(stream)
                    ids = self.select(bank, forecast_query, budget)
                    keys, values = self.fetch(bank, ids)
                    event = torch.cuda.Event()
                    event.record(stream)
                    return ids, keys, values, event

            bank['pending'] = self.pool.submit(forecast)
        self.stats['speculative_recalls'] += 1
