"""WorldTrace-Landmark: independent, benchmark-agnostic TF-A memory."""

METHOD = "worldtrace-landmark"
METADATA = {
    "training_class": "TF-A",
    "selection_type": "scene-transition",
    "memory_type": "episodic-landmark",
    "position_policy": "slot-rank-virtual-temporal",
    "eviction": "fifo",
    "attention_backend": "native-fa2",
}

