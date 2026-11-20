"""Benchmark architectures and native loader capabilities."""

from dataclasses import asdict, dataclass

from draftkv.registry import Registry, algorithm_package
from .contracts import BackboneError, InputMode


@dataclass(frozen=True)
class ModelSpec:
    backbone: str
    family: str
    task: str
    layers: int
    heads: int
    head_dim: int
    window: int
    sink: int
    chunk: int
    native_sp: tuple[int, ...]
    draftkv_sp: tuple[int, ...]
    checkpoint_id: str
    frame_tokens: int | None = None
    spatial_grid: tuple[int, int] | None = None

    def sp_status(self, sp: int, algorithm: str) -> str:
        supported = self.draftkv_sp if algorithm == "draftkv" else self.native_sp
        return "STATIC_CAPABLE_GPU_UNVERIFIED" if type(sp) is int and sp in supported else "UNSUPPORTED"


MODELS = {
    "SolarWM33B": ModelSpec("SolarWM33B", "solarwm", "minimax-h3-33b", 50, 56, 128,
                            30, 0, 5, (1, 2, 4, 8), (1, 2, 4, 8),
                            "junchaoh-cs/SolarWM-H3-33B", 1008, (24, 42)),
    "LongLive1.3B": ModelSpec("LongLive1.3B", "longlive", "t2v", 30, 12, 128,
                              12, 3, 3, (1, 2, 4, 8), (1, 2, 4, 8), "LongLive/base+LoRA"),
    "LingBot1.3B": ModelSpec("LingBot1.3B", "lingbot", "i2v-1.3B", 30, 12, 128,
                             18, 6, 4, (1, 2, 4, 8), (1, 2, 4, 8), "robbyant/lingbot-world-v2-1.3b-causal-fast"),
    "LingBot14B": ModelSpec("LingBot14B", "lingbot", "i2v-A14B", 40, 40, 128,
                            18, 6, 4, (1, 2, 4, 8), (1, 2, 4, 8), "robbyant/lingbot-world-v2-14b-causal-fast"),
}
ROUTE_MODES = {("bench1", "LongLive1.3B"): InputMode.BENCH1_T2V,
               ("bench2", "LingBot1.3B"): InputMode.ORBIA,
               ("bench2", "LingBot14B"): InputMode.ORBIA,
               ("bench2", "SolarWM33B"): InputMode.ORBIA}
ALGORITHM_REQUIREMENTS = {
    "deep-forcing": ("deep_sink", "participative_token_compression", "temporal_realignment", "denoising_reuse", "clean_commit", "native_fa2"),
    "future-forcing": ("canonical_observed_query_proxy", "historical_future_softmax", "future_profile_merge", "temporal_realignment", "clean_commit", "native_fa2"),
    "native": ("official model configuration", "native_fa2"),
    "draftkv": ("draftattention2_postrope_spatial8x16_meanmax", "hierarchical_v1",
                "original_b01_producer_plan", "asynchronous_h2d",
                "cuda_frontier", "streaming_fa2", "clean_commit", "chrono_compact_temporal", "chronological_frame_packing"),
    "shadowkv": ("author_low_rank_keys", "landmark_outliers", "sparse_rope", "clean_commit", "native_fa2"),
    "infinigen": ("author_partial_weights", "previous_layer_query_forecast", "dynamic_frame_selection",
                  "clean_commit", "native_fa2"),
    "infllm": ("author_block_representatives", "position_mapping", "author_attention_normalization",
               "clean_commit", "native_fa2"),
    "freekv": ("author_center_plus_minus_mad", "author_correction", "frame_aggregated_retrieval",
               "native_multihead_frame_fa2", "legacy_original_raft_page_available",
               "clean_commit", "native_fa2"),
    "worldkv": ("author_camera_provenance", "author_compression", "author_native_position_mapping", "clean_commit", "native_fa2"),
    "worldtrace-landmark": ("temporal_only_canonical_keys", "scene_entry_detector",
                            "frozen_landmark_fifo", "slot_rank_virtual_temporal",
                            "whole_frame_budget", "clean_commit", "native_fa2"),
}


from draftkv.algorithms.draftkv.block_policy import PROTECTED_RECENT_BY_BACKBONE, configured_protection

BLOCK_DRAFTKV_BACKBONES = frozenset(PROTECTED_RECENT_BY_BACKBONE)


def algorithm_requirements(algorithm, backbone):
    requirements = ALGORITHM_REQUIREMENTS[algorithm]
    if algorithm == 'draftkv' and backbone in BLOCK_DRAFTKV_BACKBONES:
        replace = {'hierarchical_v1': f'direct_block_topk_p{configured_protection(backbone)}',
                   'chronological_frame_packing': 'chronological_source_token_packing'}
        return tuple(replace.get(value, value) for value in requirements) + ('all_history_softmax',)
    return requirements


def model_spec(backbone: str) -> ModelSpec:
    try:
        return MODELS[backbone]
    except KeyError as exc:
        raise BackboneError("UNSUPPORTED", f"unknown backbone {backbone}") from exc


def coverage() -> dict:
    """Deterministic repository capability inventory; never an E2E attestation."""
    rows = []
    for group in Registry().groups:
        spec = model_spec(group.backbone)
        mode = ROUTE_MODES[(group.benchmark, group.backbone)]
        blockers = []
        integrated = group.algorithm in ALGORITHM_REQUIREMENTS
        if not integrated:
            blockers.append("BLOCKED_ALGORITHM_SEMANTICS")
        legacy_v1 = group.algorithm == "draftkv"
        rows.append({**group.to_dict(), "input_mode": mode.value, "family": spec.family,
                     "adapter": "draftkv.backbones.runtime:create",
                     "algorithm_adapter": "draftkv.backbones.runtime:create_algorithm",
                     "algorithm_plugin": (f"src/draftkv/algorithms/{algorithm_package(group.algorithm)}/entry.py" if group.algorithm != "native" and integrated else None),
                     "core_loader": f"draftkv.backbones.models:{spec.family.title()}Core",
                     "consumer": "original_b01_streaming_fa2" if group.algorithm == "draftkv" else "native_fa2",
                     "source_lock": "configs/backbones/source_locks.json#" + group.backbone,
                     "algorithm_source_inventory": "configs/backbones/algorithm_sources.json",
                     "inherited_evidence": "configs/backbones/evidence.json",
                     "input_capability_audit": "configs/backbones/input_capability_audit.json",
                     "native_entry": "WanI2VCausal.generate" if spec.family == "lingbot" else "LongliveCore.generate -> text-only native Forward/scheduler/clean-commit/decode",
                     "input_protocol": "SHA-frozen release_candidate or diagnostic request + exact manifest row + conditioning binding",
                     "input_conversion": None,
                     "camera_protocol": None,
                     "frozen_native_outputs": "configs/backbones/bench1_native_outputs.json" if group.benchmark == "bench1" else None,
                     "input_preparation": "scripts/bench1_prepare_inputs.py" if group.benchmark == "bench1" else "scripts/bench2_prepare_inputs.py",
                     "output_spec": {"layout": "C,F,H,W", "container": "mp4", "fps": 16,
                                     "dimensions": "frozen request; actual ffprobe required",
                                     "frame_policy": "complete C4 latent chunks; explicit frozen padding/trim only"}
                     if spec.family == "lingbot" else {"layout": "C,F,H,W", "container": "mp4", "fps": 16,
                     "frame_policy": "source-frozen text-only C3/latent240/native full decode957 at16fps; successful native output receipt; no TI parameter borrowing",
                     "input_binding": "native_text_only", "binding_gpu_acceptance": "PENDING"},
                     "hook_requirements": list(algorithm_requirements(group.algorithm, group.backbone)),
                     "sp": {str(sp): spec.sp_status(sp, group.algorithm) for sp in (1, 2, 4, 8)},
                     "implementation": "canonical_d2_postrope_spatial8x16_hierarchical_v1_chronocompact_b01_hook" if group.algorithm == "draftkv"
                     else ("author_native" if group.algorithm == "native" else ("audited_infllm_hook" if group.algorithm == "infllm" else ("frozen_original_author_hook" if integrated else "external_plugin_required"))),
                     "launchable_with_external_assets": not blockers,
                     "max_latent_frames": None,
                     "history_capacity_policy": "whole_request_runtime_capacity_no_preflight_gate" if legacy_v1 else None,
                     "full_length_static_capable": not blockers,
                     "full_dataset_launchable": False,
                     "full_dataset_blockers": blockers + ["BLOCKED_DATASET_BINDING"],
                     "launchable_here": False,
                     "sp_scope": "MODEL_TRANSPORT_CAPABILITY_NOT_ALGORITHM_ACCEPTANCE",
                     "full_length_blockers": blockers.copy(),
                     "repository_status": blockers[0] if blockers else "BLOCKED_EXTERNAL_ASSETS",
                     "blockers": blockers + ["BLOCKED_EXTERNAL_ASSETS"],
                     "runtime_registration": "NATIVE" if group.algorithm == "native" else ("CANONICAL_HOOK" if integrated else "BLOCKED_MISSING_HOOK"),
                     "missing_hook": None if integrated else {"shadowkv": "author low-rank basis/outlier mapping/sparse RoPE/clean owner", "infinigen": "pre-FSDP partial weights/previous-layer query forecast/clean owner", "worldkv": "camera PRoPE/author compression C3 vs C4/clean owner", "freekv": "strict original RAFT binary/exact ties/Center+-MAD correction/clean owner"}.get(group.algorithm),
                     "gpu_evidence": "PRE_SPATIAL8X16_EVIDENCE_ONLY" if group.algorithm == "draftkv" else "NOT_CLAIMED",
                     "release_e2e": False})
    from .budgets import author_budget_policy, effective_budget
    for row in rows:
        if row["algorithm"] == "draftkv":
            row["attention_position_policy"] = "chrono_compact"
            row["packing_policy"] = "chronological_frame_id"
            row["retrieval_position_policy"] = "native_post_rope"
            row["importance_partition"] = "spatial2d_fixed8x16"
            row["importance_pool_hw"] = [8, 16]
            row["selection_policy"] = {"protect_recent": 3,
                                       "warmup": "min(3, actual_budget)",
                                       "salient_z": 3.0, "relative_floor": 1.5}
        if row["algorithm"] == "draftkv" and row["backbone"] in BLOCK_DRAFTKV_BACKBONES:
            protected = configured_protection(row["backbone"])
            row.update(implementation=f"canonical_d2_postrope_block_topk_p{protected}_chronocompact_b01_hook",
                       packing_policy="chronological_frame_id", packing_unit="source_token",
                       gpu_evidence="BLOCK_TOPK_CPU_VALIDATED_GPU_UNVERIFIED")
            row["selection_policy"] = dict(
                protect_recent=protected, warmup=f"min({protected}, actual_budget)",
                name=f"direct_block_topk_p{protected}", remote_exclusion_gap=0,
                softmax_domain="all_nonsink_history_including_protected",
                ranking="raw_block_probability_mean_over_queries_and_heads",
                remote_token_budget=f"(actual_history_budget-min({protected},actual_history_budget))*frame_tokens",
                tail_policy="crop_last_ranked_block_to_exact_real_token_budget",
                temporal_rule="packed_token_offset//frame_tokens")
        # Keep generated route metadata in sync with frozen canonical A2:
        # the prior generated catalog still described Block-only defaults
        # for camera-conditioned backbones and mislabeled text-only LongLive.
        if row["algorithm"] == "draftkv":
            if row["backbone"] == "LongLive1.3B":
                row["implementation"] = (
                    "canonical_d2_postrope_block_topk_p3_chronocompact_b01_textonly_camera_unavailable")
            elif row["benchmark"] == "bench2" and row["backbone"] in ("LingBot1.3B", "SolarWM33B"):
                top = 3 if row["backbone"] == "LingBot1.3B" else 7
                row["implementation"] = (
                    f"a2_frustum_v2_camera_top{top}_no_early_fix_direct_block_importance_b01")
                row["selection_policy"].update(
                    camera_top_chunks=top, camera_gate_mode="hard_top_m_no_backfill",
                    softmax_domain=f"camera_top{top}_eligible_chunks_plus_recent_P_no_backfill",
                    cold_start_fallback=False, sink_tail_backfill=False,
                    temporal_floor=False)
        if row["algorithm"] == "worldtrace-landmark":
            from draftkv.algorithms.worldtrace_landmark import METADATA
            row.update(METADATA, implementation="independent_worldtrace_landmark_inference_adaptation")
        row["effective_budget"] = effective_budget(MODELS[row["backbone"]], row["algorithm"]).to_dict()
        row["author_budget_policy"] = author_budget_policy(MODELS[row["backbone"]], row["algorithm"])
        if row["backbone"] == "SolarWM33B":
            row.update(core_loader="draftkv.backbones.solarwm:SolarwmCore",
                       source_lock="configs/backbones/solarwm_source_lock.json",
                       native_entry="h3_sgf_rollout -> native decode_write",
                       camera_protocol="native H3 first-frame-relative W2C / MM-RoPE / Camera PRoPE",
                       output_spec={"layout": "C,F,H,W", "container": "mp4", "fps": 16,
                                    "frame_policy": "native C5 rollout110/decode107/362 frames, trim5 to official357"},
                       gpu_evidence="NOT_EXECUTED")
    return {"schema_version": 1, "abi_version": 1, "groups": rows,
            "models": {name: {**asdict(spec), "native_sp": list(spec.native_sp),
                              "draftkv_sp": list(spec.draftkv_sp),
                              "spatial_grid": list(spec.spatial_grid) if spec.spatial_grid else None} for name, spec in MODELS.items()}}
