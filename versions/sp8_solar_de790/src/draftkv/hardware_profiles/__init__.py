"""Offline hardware contracts; no CUDA initialization and no algorithm fallback."""

from .profiles import (ProfileError, bind_frozen, digest, load_catalog, load_profile, recipes_for,
                       seal, select_profile, source_digest, verify_binding)

__all__ = ["ProfileError", "bind_frozen", "digest", "load_catalog", "load_profile", "recipes_for",
           "seal", "select_profile", "source_digest", "verify_binding"]
