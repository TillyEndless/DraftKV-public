"""ORBIA camera/request adaptation for the native H3 image/text pipeline."""

from .solarwm_geometry import camera_arrays, frame_policy, POSITION_POLICY


def arrays(item):
    import numpy as np

    dimensions = item.document["intrinsics"]
    views, K, _ = camera_arrays(
        item.poses_c2w, item.intrinsics, dimensions["width"], dimensions["height"]
    )
    ids = frame_policy()["camera_indices"]
    return views, K, np.asarray(item.timestamps_s)[ids].copy()


def configure(request):
    policy = frame_policy()
    request.update(
        width=1344,
        height=768,
        output_width=1344,
        output_height=768,
        native_frames=policy["native_frames"],
        latent_frames=policy["latent_frames"],
        decode_latents=policy["decode_latents"],
        padding_policy=policy["padding_policy"],
        native_chunk_policy=policy["native_chunk_policy"],
        frame_policy=policy,
        num_frame_per_block=5,
        local_attn_size=30,
        sink_size=0,
        sample_shift=12.0,
        camera_coordinate_system="h3_first_frame_relative_w2c",
        camera_policy="native_h3_cadence_logd4_fixed_K",
        position_policy=POSITION_POLICY,
    )
    return request


def validate(request, source, root):
    import numpy as np
    from draftkv.benchmarks.datasets import DatasetError, sha256_file, unique_json
    from draftkv.benchmarks.orbia_requests import _resolve

    doc = unique_json((source / "case.json").read_bytes())
    with np.load(source / "camera.npz", allow_pickle=False) as archive:
        poses, times = archive["poses_c2w"], archive["timestamps_s"]
    dims = doc["intrinsics"]
    views, K, _ = camera_arrays(poses, dims["K"], dims["width"], dims["height"])
    policy = frame_policy()
    expected = configure(dict(request))
    for name in (
        "width",
        "height",
        "output_width",
        "output_height",
        "native_frames",
        "latent_frames",
        "decode_latents",
        "padding_policy",
        "native_chunk_policy",
        "frame_policy",
        "num_frame_per_block",
        "local_attn_size",
        "sink_size",
        "sample_shift",
        "camera_coordinate_system",
        "camera_policy",
        "position_policy",
    ):
        if expected[name] != request.get(name):
            raise DatasetError("H3 native request geometry differs: " + name)
    if (request.get("frames"), request.get("fps")) != (357, 16):
        raise DatasetError("H3 ORBIA requires 357 frames at official 16 FPS")
    camera = _resolve(request["camera_dir"], root)
    expected_arrays = {
        "poses": views,
        "intrinsics": K,
        "timestamps_s": times[policy["camera_indices"]],
    }
    for name, array in expected_arrays.items():
        path = camera / (name + ".npy")
        if not np.array_equal(np.load(path, allow_pickle=False), array):
            raise DatasetError(
                "H3 camera differs from deterministic native conversion: " + name
            )
        if name != "timestamps_s" and sha256_file(path) != request["asset_sha256"].get(
            name
        ):
            raise DatasetError("H3 camera hash differs: " + name)
    if (
        request.get("original_timestamps_s") != times.tolist()
        or request.get("original_intrinsics_K") != dims["K"]
        or request.get("original_events") != doc["events"]
    ):
        raise DatasetError("H3 request lost original ORBIA provenance")
    return True
