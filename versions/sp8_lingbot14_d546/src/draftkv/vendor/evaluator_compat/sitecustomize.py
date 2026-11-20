"""Opt-in interpreter import bridge for removed torchvision.io.write_video."""

import os
import sys


if os.environ.get('EVAL_COMPAT_ENABLE') == '1':
    import torchvision.io
    from torchvision_legacy_video import write_video

    if not hasattr(torchvision.io, 'write_video'):
        torchvision.io.write_video = write_video
    overlay = os.environ.get('EVAL_COMPAT_OVERLAY')
    if overlay:
        if not os.path.isdir(overlay):
            raise RuntimeError('VBench compatibility dependency overlay missing')
        sys.path.append(overlay)

    if os.environ.get('DRAFTKV_VBENCH_OFFLINE_HUB') == '1':
        from pinned_hub import install
        install()
