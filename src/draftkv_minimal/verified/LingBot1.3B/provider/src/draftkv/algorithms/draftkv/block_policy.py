"""Single source of truth for the canonical model-specific Direct Block Top-K policy.

P is the number of *complete* recent historical latent frames protected from
block ranking. It is not a different scoring algorithm or a capacity override.
"""

PROTECTED_RECENT_BY_BACKBONE = {
    "LingBot14B": 3,
    "LingBot1.3B": 3,
    "LongLive1.3B": 3,
    "SolarWM33B": 5,
}


def configured_protection(backbone):
    from .experiment import active_override
    override = active_override(backbone)
    if override is not None:
        return override['protect_recent_frames']
    try:
        return PROTECTED_RECENT_BY_BACKBONE[backbone]
    except KeyError as error:
        raise ValueError(f"unknown Block Top-K backbone: {backbone!r}") from error


def applied_protection(backbone, history_budget):
    if type(history_budget) is not int or history_budget < 1:
        raise ValueError("DRAFTKV_INVALID_HISTORY_BUDGET")
    return min(configured_protection(backbone), history_budget)
