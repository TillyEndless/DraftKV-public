"""Read-only observer of WBench's unmodified DA3 reprojection and PSNR.

The official WBench _compute_metrics is called verbatim. This module observes
the rendered/target/mask tensors *after* its native projection operation.
The additional metrics use visibility-normalized masked SSIM and masked
spatial LPIPS. They are NOT ordinary full-image SSIM or LPIPS.
"""
from __future__ import annotations

import inspect
import sys
from typing import Any

import cv2
import numpy as np

SSIM_WINDOW = 11
SSIM_SIGMA = 1.5
SSIM_MIN_WINDOW_COVERAGE = 0.6
LPIPS_PATCH = 128
LPIPS_MIN_PATCH_COVERAGE = 0.6
MIN_OFFICIAL_PIXELS = 100

# Bump the binding whenever the additional metric protocol changes.
from draftkv.evaluators.table3_reprojection import PROTOCOL_ID


def _mean(values):
    return float(np.mean(values)) if values else None


def _masked_ssim(left: np.ndarray, right: np.ndarray, mask: np.ndarray):
    """Gaussian moments normalized by valid-pixel mass, excluding sparse windows.

    For an all-valid mask this reduces to gaussian-window SSIM with population
    variances (C1=0.01², C2=0.03²). No invalid RGB value enters a moment.
    """
    h, w = mask.shape
    if h < SSIM_WINDOW or w < SSIM_WINDOW:
        return None, 0
    mask_f = mask.astype(np.float32)
    support = cv2.boxFilter(mask_f, -1, (SSIM_WINDOW, SSIM_WINDOW),
                            normalize=True, borderType=cv2.BORDER_CONSTANT)
    allowed = (support >= SSIM_MIN_WINDOW_COVERAGE) & (mask_f > 0)
    pad = SSIM_WINDOW // 2
    allowed[:pad, :] = False
    allowed[-pad:, :] = False
    allowed[:, :pad] = False
    allowed[:, -pad:] = False
    count = int(allowed.sum())
    if count == 0:
        return None, 0

    def blur(x):
        return cv2.GaussianBlur(x, (SSIM_WINDOW, SSIM_WINDOW), SSIM_SIGMA,
                                borderType=cv2.BORDER_CONSTANT)

    z = blur(mask_f)
    norm = np.maximum(z, 1e-8)
    per_channel = []
    for c in range(3):
        a = left[:, :, c].astype(np.float32)
        b = right[:, :, c].astype(np.float32)
        ma, mb = blur(mask_f * a) / norm, blur(mask_f * b) / norm
        va = np.maximum(blur(mask_f * a * a) / norm - ma * ma, 0.)
        vb = np.maximum(blur(mask_f * b * b) / norm - mb * mb, 0.)
        cov = blur(mask_f * a * b) / norm - ma * mb
        numerator = (2 * ma * mb + 0.01 ** 2) * (2 * cov + 0.03 ** 2)
        denominator = (ma * ma + mb * mb + 0.01 ** 2) * (va + vb + 0.03 ** 2)
        per_channel.append(numerator / np.maximum(denominator, 1e-12))
    similarity = np.mean(np.stack(per_channel, axis=0), axis=0)
    return float(similarity[allowed].mean()), count


class ReprojectionObserver:
    """Supplementary metrics on the exact same WBench reprojection RGB/mask."""

    def __init__(self, device: str = "cuda", lpips_model=None):
        self.device = device
        self._lpips_model = lpips_model
        self.frames: list[dict[str, Any]] = []

    def _load_lpips(self):
        if self._lpips_model is None:
            import lpips
            self._lpips_model = lpips.LPIPS(
                net="alex", version="0.1", spatial=True).to(self.device).eval()
        return self._lpips_model

    def _masked_lpips(self, left, right, mask):
        """Mask-weighted LPIPS spatial maps over adequately covered 128² regions.

        At undefined pixels both inputs are set to the SAME target RGB. This is
        merely a neutral input for the feature extractor, never a changed WBench
        render, an inferred correspondence, interpolation or geometric inpaint.
        The final score aggregates ONLY valid-pixel spatial LPIPS responses.
        """
        h, w = mask.shape
        patches = []
        for y in range(0, h - LPIPS_PATCH + 1, LPIPS_PATCH):
            for x in range(0, w - LPIPS_PATCH + 1, LPIPS_PATCH):
                region = (slice(y, y + LPIPS_PATCH), slice(x, x + LPIPS_PATCH))
                m = mask[region].astype(bool)
                if float(m.mean()) >= LPIPS_MIN_PATCH_COVERAGE:
                    patches.append((region, m))
        if not patches:
            return None, 0, 0
        import torch
        model = self._load_lpips()
        l_batch, r_batch, masks = [], [], []
        for region, m in patches:
            source = left[region].copy()
            target = right[region].copy()
            source[~m] = target[~m]
            l_batch.append(source)
            r_batch.append(target)
            masks.append(m)
        # Process patches in small batches to bound VRAM for long videos.
        weighted_total = 0.
        weight_total = 0.
        with torch.inference_mode():
            for start in range(0, len(patches), 4):
                sl = slice(start, start + 4)
                a = torch.from_numpy(np.stack(l_batch[sl])).permute(0, 3, 1, 2).to(self.device)
                b = torch.from_numpy(np.stack(r_batch[sl])).permute(0, 3, 1, 2).to(self.device)
                weights = torch.from_numpy(np.stack(masks[sl]).astype(np.float32))
                weights = weights[:, None].to(self.device)
                result = model(a * 2 - 1, b * 2 - 1)
                if result.ndim != 4 or result.shape[-2:] != weights.shape[-2:]:
                    raise ValueError("LPIPS backend must expose a full-resolution spatial map")
                weighted_total += float((result * weights).sum().item())
                weight_total += float(weights.sum().item())
        return (weighted_total / weight_total if weight_total > 0 else None), len(patches), int(weight_total)

    def observe(self, index, rendered, target, valid):
        mask = valid.detach().to("cpu").numpy().astype(np.uint8)
        h, w = mask.shape
        valid_pixels = int(mask.sum())
        row: dict[str, Any] = {
            "frame_index": int(index),
            "rendered_pixels": valid_pixels,
            "pixel_coverage": valid_pixels / (h * w),
            "ssim": None,
            "ssim_valid_centers": 0,
            "ssim_window_coverage": 0.,
            "lpips": None,
            "lpips_valid_patches": 0,
            "lpips_candidate_patches": (h // LPIPS_PATCH) * (w // LPIPS_PATCH),
            "lpips_patch_coverage": 0.,
            "lpips_valid_pixels": 0,
            "reason": None,
        }
        if valid_pixels < MIN_OFFICIAL_PIXELS:
            row["reason"] = "BELOW_WBENCH_MINIMUM_RENDERED_PIXELS"
            self.frames.append(row)
            return

        # Only local metric input copies: official PSNR tensors stay unchanged.
        left = rendered.detach().to("cpu").numpy().astype(np.float32) / 255.
        right = target.detach().to("cpu").numpy().astype(np.float32) / 255.
        row["ssim"], count = _masked_ssim(left, right, mask)
        row["ssim_valid_centers"] = count
        row["ssim_window_coverage"] = count / (h * w)
        row["lpips"], count_patches, count_pixels = self._masked_lpips(left, right, mask)
        row["lpips_valid_patches"] = count_patches
        row["lpips_valid_pixels"] = count_pixels
        row["lpips_patch_coverage"] = (
            count_patches / row["lpips_candidate_patches"]
            if row["lpips_candidate_patches"] else 0.)
        self.frames.append(row)

    def run(self, official_module, video_path, da3_dir, *, fps=3.0, max_pts_per_frame=5000):
        """Call the upstream compute_case intact, observing its original locals."""
        target = official_module._compute_metrics
        source, start = inspect.getsourcelines(target)
        anchors = [start + idx for idx, line in enumerate(source)
                   if line.strip() == "if has_render.sum() < 100:"]
        if len(anchors) != 1 or not any(
                line.strip() == "has_render = z_buf < float('inf')" for line in source):
            raise RuntimeError("Official WBench instrumentation anchor changed")
        anchor = anchors[0]
        previous = sys.gettrace()
        error_messages = []

        def local_trace(frame, event, arg):
            if event == "line" and frame.f_lineno == anchor:
                v = frame.f_locals
                try:
                    self.observe(v["i"], v["rendered"],
                                 v["frames_t"][v["i"]], v["has_render"])
                except Exception as exc:
                    error_messages.append(f"frame={v.get('i')}: {type(exc).__name__}: {exc}")
                    self.frames.append({"frame_index": int(v.get("i", -1)),
                                        "ssim": None, "lpips": None,
                                        "error": error_messages[-1]})
            return local_trace

        def tracer(frame, event, arg):
            if event == "call" and frame.f_code is target.__code__:
                return local_trace
            return None

        if previous is not None:
            raise RuntimeError("Cannot instrument WBench with another active tracer")
        try:
            sys.settrace(tracer)
            official_result = official_module.compute_case(
                str(video_path), str(da3_dir), fps=fps, device=self.device,
                max_pts_per_frame=max_pts_per_frame)
        finally:
            sys.settrace(previous)
        if official_result.get("error"):
            raise RuntimeError(f"WBench official reprojection failed: {official_result['error']}")
        if error_messages:
            raise RuntimeError("Supplementary similarity failed: " + "; ".join(error_messages[:3]))
        expected = int(official_result["details"]["num_frames"])
        indices = [r["frame_index"] for r in self.frames]
        if len(indices) != len(set(indices)) or any(i < 0 or i >= expected for i in indices):
            raise RuntimeError(f"WBench trace has duplicate/out-of-range frame indices: {indices}")
        # Official WBench can early-continue when no historical points exist.
        # Unvisited frames carry zero support, not a fabricated similarity.
        records = {r["frame_index"]: r for r in self.frames}
        self.frames = [records.get(i, {"frame_index": i, "ssim": None,
                       "lpips": None, "rendered_pixels": 0,
                       "pixel_coverage": 0., "reason": "NO_REPROJECTION_CANDIDATES"})
                       for i in range(expected)]
        scores = {
            "reprojection_psnr": float(official_result["details"]["photometric_psnr"]),
            "reprojection_ssim": _mean([r["ssim"] for r in self.frames if r["ssim"] is not None]),
            "reprojection_lpips": _mean([r["lpips"] for r in self.frames if r["lpips"] is not None]),
        }
        counts = {
            "psnr_eligible_frames": sum(r.get("rendered_pixels", 0) >= MIN_OFFICIAL_PIXELS
                                         for r in self.frames),
            "ssim_frames": sum(r["ssim"] is not None for r in self.frames),
            "lpips_frames": sum(r["lpips"] is not None for r in self.frames),
        }
        if scores["reprojection_psnr"] <= 0:
            scores["reprojection_psnr"] = None
        return {
            "scores": scores,
            "counts": counts,
            "frames": self.frames,
            "wbench_official": official_result,
            "protocol": {
                "id": PROTOCOL_ID,
                "geometry": "Pinned WBench _compute_metrics unmodified",
                "fps": fps, "max_pts_per_frame": max_pts_per_frame,
                "ssim": "gaussian moments normalized by WBench visibility mask",
                "ssim_window": SSIM_WINDOW, "ssim_sigma": SSIM_SIGMA,
                "ssim_min_window_coverage": SSIM_MIN_WINDOW_COVERAGE,
                "lpips": "visibility-weighted spatial LPIPS; not full-image LPIPS",
                "lpips_net": "alex-v0.1", "lpips_patch": LPIPS_PATCH,
                "lpips_min_patch_coverage": LPIPS_MIN_PATCH_COVERAGE,
                "undefined_pixel_policy": "both feature inputs identical at invalid mask pixels",
            },
        }
