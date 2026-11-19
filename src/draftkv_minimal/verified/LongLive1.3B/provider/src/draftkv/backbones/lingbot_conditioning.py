"""Validation of explicit frozen native camera assets; no input conversion."""

from .contracts import BackboneError


def validate_static_camera(poses, intrinsics, np):
    """Check actual prepared arrays against the source policy before inference."""
    if (not np.allclose(poses, np.eye(4), rtol=0, atol=1e-6)
            or not np.allclose(intrinsics, intrinsics[:1], rtol=0, atol=1e-6)):
        raise BackboneError("BLOCKED_INPUT_PROTOCOL", "frozen TI camera policy requires stationary identity C2W and fixed intrinsics")
