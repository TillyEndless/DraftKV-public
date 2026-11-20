"""Contiguous exact frame transport shared by video comparator adapters."""


def frame_token_ids(frames, frame_tokens):
    import torch

    return (
        frames[..., None] * frame_tokens
        + torch.arange(frame_tokens, device=frames.device)
    ).flatten(-2)


def fetch_frames(archive, frames, frame_tokens, bank, stats):
    """[H,R] frame IDs -> exact [1,H,R*F,D] KV; one packed host transfer.

    Frame slices never cross a native chunk boundary. Reuse a complete resident
    selection when possible; no scattered token/page physical gather is used.
    """
    import torch

    if frames.ndim != 2 or frames.dtype != torch.long or not archive:
        raise ValueError("whole historical frames require a nonempty clean archive")
    heads, count = frames.shape
    chunk_tokens, dim = archive[0].shape[-2:]
    if chunk_tokens % frame_tokens or heads != archive[0].shape[1]:
        raise ValueError("frame/chunk/head alignment differs")
    ids = frame_token_ids(frames, frame_tokens)
    # This exact frame witness is required anyway to index the CPU archive.
    # Reuse it for validation and resident-cache equality instead of first
    # synchronizing a large CUDA token-ID equality and then copying frames.
    frames_host = frames.detach().to("cpu")
    if (
        bool((frames_host < 0).any())
        or bool((frames_host >= len(archive) * chunk_tokens // frame_tokens).any())
        or (count > 1 and bool((frames_host.sort(-1).values.diff(dim=-1) == 0).any()))
    ):
        raise ValueError("duplicate, future or negative retrieved frame")
    resident_frames = bank.get("resident_frames_host")
    if resident_frames is not None and torch.equal(frames_host, resident_frames):
        stats["reused_tokens"] += ids.numel()
        return bank["resident_k"][None], bank["resident_v"][None]
    packed = torch.empty(
        (2, heads, count * frame_tokens, dim),
        dtype=archive[0].dtype,
        device="cpu",
        pin_memory=frames.is_cuda,
    )
    for head, row in enumerate(frames_host.tolist()):
        for slot, frame in enumerate(row):
            chunk, offset = divmod(frame * frame_tokens, chunk_tokens)
            packed[:, head, slot * frame_tokens : (slot + 1) * frame_tokens].copy_(
                archive[chunk][:, head, offset : offset + frame_tokens]
            )
    packed = packed.to(frames.device, non_blocking=frames.is_cuda)
    bank.update(resident_ids=ids.clone(), resident_frames_host=frames_host,
                resident_k=packed[0], resident_v=packed[1])
    stats["fetched_value_tokens"] += ids.numel()
    return packed[0][None], packed[1][None]
