"""Source-locked B01 deployment bridge. CUDA is loaded only on explicit execution."""
from .config import consumer_for, frame_frontiers, transfer_schedule, select_config

__all__ = ['consumer_for', 'frame_frontiers', 'transfer_schedule', 'select_config']
