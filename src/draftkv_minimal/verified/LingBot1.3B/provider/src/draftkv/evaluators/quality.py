"""Official VBench normalization, applied to explicitly labelled Long custom input."""
from __future__ import annotations
import math

UPSTREAM_COMMIT = 'fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490'
QUALITY_SCORE_VERSION = 'vbench:' + UPSTREAM_COMMIT + ':quality-7d-percent/long-custom-dev-v1'
OFFICIAL_QUALITY_PROVENANCE = {
    'repository': 'https://github.com/Vchitect/VBench', 'commit': UPSTREAM_COMMIT,
    'scripts/constant.py': 'c265be4cb51f95257e0b9dfbcb95453d39edca47ee14cb3f7b11e6f20f165e17',
    'scripts/cal_final_score.py': '1d3150e66429fde427c77168ab4c1c9839f9cd42291304cab148213de4ff28fc',
    'clipping': False, 'weight_sum': 6.5,
}
QUALITY_LABEL = 'VBench-Long Quality (7D)'
QUALITY_DIMENSIONS = ('subject_consistency', 'background_consistency', 'temporal_flickering',
                      'motion_smoothness', 'dynamic_degree', 'aesthetic_quality', 'imaging_quality')
AVG6_DIMENSIONS = tuple(d for d in QUALITY_DIMENSIONS if d != 'temporal_flickering')
# scripts/constant.py NORMALIZE_DIC and DIM_WEIGHT at UPSTREAM_COMMIT.
NORMALIZATION = {
    'subject_consistency': (0.1462, 1.0, 1.0),
    'background_consistency': (0.2615, 1.0, 1.0),
    'temporal_flickering': (0.6293, 1.0, 1.0),
    'motion_smoothness': (0.706, 0.9975, 1.0),
    'dynamic_degree': (0.0, 1.0, 0.5),
    'aesthetic_quality': (0.0, 1.0, 1.0),
    'imaging_quality': (0.0, 1.0, 1.0),
}


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def quality_fields(scores, *, failed=False):
    """No clipping, imputation or reweighting. Avg6 always uses the six raw scores."""
    scores = scores or {}
    complete = all(finite(scores.get(d)) for d in QUALITY_DIMENSIONS)
    avg6 = (sum(scores[d] for d in AVG6_DIMENSIONS) / 6
            if all(finite(scores.get(d)) for d in AVG6_DIMENSIONS) else None)
    # Match upstream get_nomalized_score/get_quality_score operation order.
    order = ('subject_consistency', 'background_consistency', 'temporal_flickering',
             'motion_smoothness', 'aesthetic_quality', 'imaging_quality', 'dynamic_degree')
    quality = normalized = None
    if complete:
        weighted = [(scores[d] - NORMALIZATION[d][0]) /
                    (NORMALIZATION[d][1] - NORMALIZATION[d][0]) * NORMALIZATION[d][2] for d in order]
        normalized = sum(weighted) / sum(NORMALIZATION[d][2] for d in order)
        quality = 100 * sum(weighted) / sum(NORMALIZATION[d][2] for d in order)
        if not math.isfinite(quality):
            complete, quality, normalized, failed = False, None, None, True
    return dict(quality_score=quality, vbench_quality_score_percent=quality,
                vbench_quality_score_normalized=normalized,
                quality_score_provenance=OFFICIAL_QUALITY_PROVENANCE, quality_score_status='complete' if complete else 'failed' if failed else 'incomplete',
                vbench_long_avg6=avg6, quality_score_version=QUALITY_SCORE_VERSION, quality_score_label=QUALITY_LABEL)


def full_benchmark_cohort(backbone, dataset_sha256, case_ids, scope):
    """Authenticate formal cohort size/order/pin independently of scoring success."""
    from draftkv.benchmarks.bench1 import MANIFEST_PINS, formal_scope
    from draftkv.benchmarks.datasets import Backbone
    from draftkv.benchmarks.worldattention import CASE_IDS
    try:
        model = Backbone(backbone.lower())
    except ValueError:
        return False
    if model not in MANIFEST_PINS or dataset_sha256 != MANIFEST_PINS[model] or scope != formal_scope(backbone):
        return False
    expected = list(CASE_IDS)
    return case_ids == expected


def cohort_quality_fields(scores, *, formal, failed=False):
    result = quality_fields(scores, failed=failed)
    result['quality_score_scope'] = 'full-benchmark' if formal else 'exploratory-subset'
    result['formal_quality_score_percent'] = result['vbench_quality_score_percent'] if formal else None
    result['formal_quality_score_normalized'] = result['vbench_quality_score_normalized'] if formal else None
    return result
