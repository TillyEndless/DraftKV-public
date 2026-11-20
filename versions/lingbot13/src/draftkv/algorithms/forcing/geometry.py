"""Source-derived output accounting; never resize or alter native inputs."""
from draftkv.backbones.contracts import BackboneError
from .budget import METHODS


def native_geometry(plan):
    if plan["backbone"] == "LingBot14B":
        # Reuse the existing source/image-locked NumPy floor calculation,
        # including its 480 -> 464 floating-point boundary behavior.
        from draftkv.backbones.acceptance import native_capture_geometry
        return native_capture_geometry(plan)
    if plan["backbone"] == "SolarWM33B":
        # H3 native latent grid is [48,84] and attention downsamples it again
        # to [24,42] = 1008 tokens per frame. H/16 x W/16 is the VAE latent
        # grid (4032), NOT the H3 attention grid. Reuse the source-derived
        # capture contract rather than inferring attention shape from pixels.
        from draftkv.backbones.acceptance import native_capture_geometry
        geometry = native_capture_geometry(plan)
        request = plan["request"]
        if [request["output_height"], request["output_width"]] != geometry["native_output_hw"]:
            raise BackboneError("BLOCKED_OUTPUT_SPEC", "H3 native frame dimensions changed")
        return geometry
    request = plan["request"]
    h, w = request["output_height"], request["output_width"]
    return dict(kind="FROZEN_NATIVE_T2V", native_output_hw=[h, w],
                frozen_output_hw=[h, w], attention_grid_hw=[h // 16, w // 16],
                frame_tokens=(h // 16) * (w // 16), output_spec_status="MATCH")


def output_request(plan):
    """Expected native decode only; the frozen generation request stays intact."""
    request = plan["request"]
    if plan["algorithm"] not in METHODS:
        return request
    geometry = plan.get("native_output_geometry")
    actual = native_geometry(plan)
    # The control and worker environments may use different NumPy releases;
    # require the same source, input and computed dimensions, not package text.
    identity = lambda value: {k: v for k, v in value.items() if k != "numpy_version"}
    if not geometry or identity(geometry) != identity(actual):
        raise BackboneError("SOURCE_ABI_DRIFT", "forcing native output geometry changed")
    h, w = geometry["native_output_hw"]
    return dict(request, output_height=h, output_width=w)
