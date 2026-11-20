"""Deterministic frame-equivalent accounting, independent of observed quality."""
from fractions import Fraction

METHODS = ("deep-forcing", "future-forcing")
FUTURE_PRESETS = {"author_primary": Fraction(18, 21),
                  "short_cr_28_57": Fraction(2, 7),
                  "short_cr_42_85": Fraction(3, 7)}
UPSTREAM = {
    "deep-forcing": {"url": "https://github.com/cvlab-kaist/DeepForcing",
                     "revision": "ea9961aa4bd5b554daeecdcf59e37c495fba9df6",
                     "paper": "arxiv:2512.05081v1"},
    "future-forcing": {"url": "https://arxiv.org/abs/2605.30083v1",
                       "revision": "arxiv:2605.30083v1",
                       "code_revision": None, "implementation": "paper_based_reproduction"},
}


def round_frame(value):
    value = Fraction(value)
    return (2 * value.numerator + value.denominator) // (2 * value.denominator)


def allocation(total, weights):
    exact = [Fraction(total * w, sum(weights)) for w in weights]
    result = [v.numerator // v.denominator for v in exact]
    order = sorted(range(len(weights)), key=lambda i: (-(exact[i] - result[i]), i))
    for i in order[:total - sum(result)]:
        result[i] += 1
    return tuple(result)


def policy_budget(window, current, method, *, preset="author_primary"):
    if (type(window) is not int or type(current) is not int
            or not 0 < current < window or method not in METHODS):
        raise ValueError("invalid forcing native window/current/method")
    history = window - current
    if method == "deep-forcing":
        if preset != "author_primary":
            raise ValueError("Deep Forcing has one release preset")
        ratio = Fraction(16, 21)
        retained = round_frame(ratio * history)
        sink, selected, recent = allocation(retained, (10, 2, 1))
        # Primary release allocations override the historical author-ratio map
        # for the explicitly recommended native window/current geometries.
        if (window, current) == (12, 3):
            sink, selected, recent = 5, 3, 1
            retained = history
            ratio = Fraction(1, 1)
        if (window, current) in ((18, 4), (30, 5)):
            sink, selected, recent = {(18, 4): (8, 2, 4), (30, 5): (13, 9, 3)}[(window, current)]
            retained = history
            ratio = Fraction(1, 1)
        target, high_water, denominator = retained + current, window, history
        original = {"window": 21, "compressed": 16, "sink": 10,
                    "selected": 2, "recent_including_current": 4, "current": 3}
        actual_ratio = Fraction(retained, history)
    else:
        if preset not in FUTURE_PRESETS:
            raise ValueError("unknown Future Forcing preset")
        ratio = FUTURE_PRESETS[preset]
        if preset == "author_primary" and (window, current) in ((18, 4), (30, 5)):
            ratio = Fraction(1, 1)
        target = round_frame(ratio * window)
        if target <= current:
            raise ValueError("frame-rounded Future Forcing capacity leaves no historical KV")
        retained, high_water, denominator = target - current, target, window
        sink, selected, recent = 0, retained, 0
        original = {"primary_retained_frames": 18, "reference_context_frames": 21,
                    "reference_context_origin": "Self_Forcing_context; declared_cross_host_convention",
                    "code_default_verified": False}
        actual_ratio = Fraction(target, window)
    # These capacities are explicit recommendations, not author-ratio results.
    recommended = (window, current) in ((18, 4), (30, 5)) and preset == "author_primary"
    error = actual_ratio - ratio
    return dict(method=method, preset=preset, unit="latent_frame_equivalent",
                native_frames=window, current_frames=current, native_history_frames=history,
                compressed_frames=target, active_history_frames=retained,
                attention_cap_frames=high_water, sink_frames=sink,
                selected_frames=selected, recent_history_frames=recent,
                retained_ratio_numerator=ratio.numerator, retained_ratio_denominator=ratio.denominator,
                ratio_applies_to="history" if method == "deep-forcing" else "total_native_capacity",
                ratio_denominator_frames=denominator, ratio_error=float(error),
                relative_ratio_error=float(error / ratio),
                rounding=("explicit_recommended_allocation_20261104" if recommended else
                          "nearest_integer_half_up; largest_remainder_ties_sink_selected_recent"),
                author_original=original, upstream=UPSTREAM[method],
                proxy_history_frames=history if method == "future-forcing" else None)
