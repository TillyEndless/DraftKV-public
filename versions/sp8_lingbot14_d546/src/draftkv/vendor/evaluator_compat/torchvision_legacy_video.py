"""Video-only torchvision 0.23 write_video compatibility, never score math."""

import numpy as np


def write_video(filename, video_array, fps, video_codec='libx264', options=None,
                audio_array=None, audio_fps=None, audio_codec=None, audio_options=None):
    if audio_array is not None or audio_fps is not None or audio_codec is not None or audio_options is not None:
        raise NotImplementedError('VBench-Long compatibility supports video-only writes')
    import av
    import torch

    frames = torch.as_tensor(video_array, dtype=torch.uint8).numpy(force=True)
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError('Expected uint8 [T,H,W,3] frames')
    if isinstance(fps, float):
        fps = int(np.round(fps))
    with av.open(filename, mode='w') as container:
        stream = container.add_stream(video_codec, rate=fps)
        stream.width = frames.shape[2]
        stream.height = frames.shape[1]
        stream.pix_fmt = 'yuv420p' if video_codec != 'libx264rgb' else 'rgb24'
        stream.options = options or {}
        for image in frames:
            frame = av.VideoFrame.from_ndarray(image, format='rgb24')
            try:
                frame.pict_type = 'NONE'
            except TypeError:
                from av.video.frame import PictureType
                frame.pict_type = PictureType.NONE
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
