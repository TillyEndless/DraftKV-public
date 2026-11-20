"""A2: exact full-resolution region-level relevance, NOT token-softmax mass.

Each mean/max logit aggregates the Cartesian query-region × historical-block
full-token QK matrix. Branch softmax is over ALL eligible historical blocks,
including protected recent frames. Ceil-edge real tokens only. No Draft pooling.
"""
from .config import AblationError

A2_NAME = 'A2: exact full-resolution region-level relevance'
A2_AVAILABLE = True


class ExactRegionRelevanceOracle:
    def __init__(self, *, available=None, chunk_tokens=128):
        if not (A2_AVAILABLE if available is None else available):
            raise AblationError('Exact region-relevance oracle is unavailable. Do not fall back to token-softmax attention mass.')
        if type(chunk_tokens) is not int or chunk_tokens < 1:
            raise AblationError('positive oracle chunk_tokens required')
        self.chunk_tokens = chunk_tokens

    def region_logits(self, query, keys, query_regions, key_blocks, *, scale=None):
        """Q/K [B,T,H,D]; maps explicitly enumerate valid full-resolution tokens.

        No dense all-history attention allocation: bound each temporary QK tile.
        Return branch logits [B,H,Q_regions,K_blocks], before production FP16 cast.
        """
        import torch
        if query.ndim != 4 or keys.ndim != 4 or query.shape[0] != keys.shape[0] or query.shape[2:] != keys.shape[2:]:
            raise AblationError('oracle Q/K shape mismatch')
        if not query_regions or not key_blocks:
            raise AblationError('oracle needs nonempty real-token regions')
        for regions, length in ((query_regions, query.shape[1]), (key_blocks, keys.shape[1])):
            flat = [x for region in regions for x in region]
            if (not all(regions) or len(flat) != len(set(flat)) or any(type(x) is not int or not 0 <= x < length for x in flat)):
                raise AblationError('oracle maps must contain unique valid tokens')
        scale = query.shape[-1] ** -.5 if scale is None else scale
        means, maxima = [], []
        for qr in query_regions:
            mean_row, max_row = [], []
            for kb in key_blocks:
                total = torch.zeros((query.shape[0], query.shape[2]), device=query.device, dtype=torch.float32)
                maximum = torch.full_like(total, float('-inf'))
                for i in range(0, len(qr), self.chunk_tokens):
                    q = query[:, qr[i:i+self.chunk_tokens]].float().transpose(1, 2)
                    for j in range(0, len(kb), self.chunk_tokens):
                        k = keys[:, kb[j:j+self.chunk_tokens]].float().transpose(1, 2)
                        logits = (q @ k.transpose(-1, -2)) * scale
                        total = total + logits.sum((-1, -2))
                        maximum = torch.maximum(maximum, logits.amax((-1, -2)))
                mean_row.append(total / (len(qr)*len(kb)))
                max_row.append(maximum)
            means.append(torch.stack(mean_row, -1))
            maxima.append(torch.stack(max_row, -1))
        return torch.stack(means, -2), torch.stack(maxima, -2)

    def probabilities(self, query, keys, query_regions, key_blocks, *, config, scale=None):
        import torch
        mean, maximum = self.region_logits(query, keys, query_regions, key_blocks, scale=scale)
        p = mean.to(torch.float16).float().softmax(-1)
        weight = float(config.maxpool_weight)
        if weight:
            p = (1-weight)*p + weight*maximum.to(torch.float16).float().softmax(-1)
        return p.to(torch.float16)

    def score(self, query, keys, h, w, *, config):
        from draftkv.retrieval.draftattention2_importance import spatial_blocks
        blocks = spatial_blocks(h, w, config.pool_h, config.pool_w)
        ft = h*w
        if query.shape[1] % ft or keys.shape[1] % ft:
            raise AblationError('oracle needs full real-token frames')
        qr = [tuple(f*ft+t for t in b) for f in range(query.shape[1]//ft) for b in blocks]
        kr = [tuple(f*ft+t for t in b) for f in range(keys.shape[1]//ft) for b in blocks]
        return self.probabilities(query, keys, qr, kr, config=config)
