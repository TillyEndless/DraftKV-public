"""H3 native geometry; no retrieval, scoring, cache or attention algorithm.

Video pages use the native MM-RoPE coordinate system with an absolute temporal
origin. For each sliding W6 call the condition prefix receives the same temporal
translation. This is a common orthogonal change of basis on Q/K (first 96
channels); camera PRoPE occupies the disjoint last 32 channels. Thus native
window logits are unchanged and clean video pages have a stable coordinate
system. No Wan rotary formula is used here.
"""

from .contracts import BackboneError


POSITION_POLICY = "h3_native_mmrope_camera_prope"


def frame_policy(frames=357):
    if type(frames) is not int or frames < 22:
        raise ValueError("H3 requires at least 22 source frames")
    n = (frames - 5 + 16) // 17
    decode = 5 * n + 2
    rollout = (decode + 4) // 5 * 5
    indices, source, tail = [0], 0, 0
    for i in range(rollout - 1):
        source += (1, 4, 4, 4, 4)[i % 5]
        if source < frames:
            indices.append(source)
        else:
            indices.append(indices[-1])
            tail += 1
    return dict(
        target_frames=frames,
        native_frames=17 * n + 5,
        latent_frames=rollout,
        decode_latents=decode,
        latent_chunk_frames=5,
        chunks=rollout // 5,
        camera_indices=indices,
        tail_camera_rows=tail,
        video_trim_frames=17 * n + 5 - frames,
        padding_policy="h3_native_decode_trim",
        native_chunk_policy="h3_source_length_c5",
    )


def camera_arrays(poses, intrinsics, width, height, *, frames=357):
    """Native first-frame-relative W2C, normalized K and native cadence.

    No interpolation, axis flip, synthetic camera or mutation of source inputs.
    The native PRoPE then applies the checkpoint's logd4/fixed-K contract.
    """
    import numpy as np

    poses = np.asarray(poses, dtype=np.float32)
    K = np.asarray(intrinsics, dtype=np.float32)
    if poses.shape != (frames, 4, 4) or K.shape != (3, 3) or min(width, height) <= 0:
        raise ValueError("invalid source camera shape")
    if (
        not np.isfinite(poses).all()
        or not np.isfinite(K).all()
        or not np.allclose(poses[:, 3], [0, 0, 0, 1], atol=1e-5)
        or not np.allclose(
            poses[:, :3, :3].transpose(0, 2, 1) @ poses[:, :3, :3], np.eye(3), atol=1e-3
        )
        or not np.allclose(np.linalg.det(poses[:, :3, :3]), 1, atol=1e-3)
        or K[0, 1] != 0
        or K[1, 0] != 0
        or min(K[0, 0], K[1, 1]) <= 0
    ):
        raise ValueError("invalid OpenCV camera or intrinsics")

    def invert(values):
        out = np.zeros_like(values)
        out[..., :3, :3] = values[..., :3, :3].swapaxes(-1, -2)
        out[..., :3, 3] = -(out[..., :3, :3] @ values[..., :3, 3, None])[..., 0]
        out[..., 3, 3] = 1
        return out

    relative = invert(poses[:1]) @ poses
    relative[0] = np.eye(4)
    views = invert(relative)
    normalized = K.copy()
    normalized[0] /= width
    normalized[1] /= height
    ids = frame_policy(frames)["camera_indices"]
    return (
        views[ids].copy(),
        np.repeat(normalized[None], len(ids), axis=0),
        poses[ids].copy(),
    )


class NativeGeometry:
    policy = POSITION_POLICY

    def __init__(self, model, control):
        from solarwm.backends.minimax_h3.torch_geometry import (
            native_video_position_grid,
        )
        from solarwm.backends.minimax_h3.model import _apply_rotary_emb

        self.control, self.rope, self.apply = control, model.rope, _apply_rotary_emb
        self.rows = int(control.layout.rows_per_video_frame)
        if (self.rows, control.layout.latent_height, control.layout.latent_width) != (
            1008,
            48,
            84,
        ):
            raise BackboneError(
                "SOURCE_ABI_DRIFT", "H3 native spatial geometry changed"
            )
        self.positions = native_video_position_grid(
            control.viewmats.shape[1] - 1,
            48,
            84,
            origin=float(control.layout.text_indices.numel()),
        ).to(control.viewmats.device)

    def sparse_rope(self, keys, ids, height, width):
        if (height, width) != (24, 42):
            raise BackboneError(
                "KV_CONSTRAINT", "H3 sparse rotary used with foreign geometry"
            )
        cos, sin = self.rope(self.positions[ids.reshape(-1)])
        cos = cos.reshape(*ids.shape, -1).to(keys.dtype)
        sin = sin.reshape(*ids.shape, -1).to(keys.dtype)
        import torch

        first, second = keys[..., :96].chunk(2, -1)
        rotated = keys[..., :96] * cos + torch.cat((-second, first), -1) * sin
        return torch.cat((rotated, keys[..., 96:]), -1)

    def retrieval_query(self, context):
        import torch

        ids = (
            torch.arange(context.query.shape[1], device=context.query.device)
            % self.rows
        )
        ids += (context.current_frames - 1) * self.rows
        return self.sparse_rope(
            context.raw_query[0].transpose(0, 1),
            ids.expand(context.query.shape[2], -1),
            24,
            42,
        ).transpose(0, 1)[None]
