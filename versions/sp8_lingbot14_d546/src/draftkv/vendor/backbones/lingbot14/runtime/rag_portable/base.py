import torch


def host_copy(tensor):
    result = torch.empty(tensor.shape, dtype=tensor.dtype, device='cpu')
    result.copy_(tensor.detach(), non_blocking=False)
    return result


def host_copy_pair(key, value):
    """P0-3A: bitwise identical direct K/V D2H, without GPU torch.stack.

    The CPU Bank stays pageable, so this does not grow indefinitely pinned.
    A pinned staging allocation permits DMA; intentionally synchronizes before
    returning to retain the original clean-commit contract.
    """
    if key.shape != value.shape or key.dtype != value.dtype:
        raise ValueError("WorldKV K/V clean pair shape or dtype differs")
    if key.device.type != "cuda":
        return torch.stack((key.detach(), value.detach())).to("cpu")
    pinned = torch.empty(
        (2, *key.shape), device="cpu", dtype=key.dtype, pin_memory=True
    )
    pinned[0].copy_(key.detach(), non_blocking=False)
    pinned[1].copy_(value.detach(), non_blocking=False)
    bank = torch.empty(pinned.shape, dtype=pinned.dtype, device="cpu")
    bank.copy_(pinned)
    return bank


class BaseCache:
    def clear(self):
        self.raw.clear()
        self.banks.clear()

    def dump(self):
        return None
