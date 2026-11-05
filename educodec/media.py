"""Decode ordinary video containers to the codec's RGB frame representation."""
import io
import numpy as np


def read_video(data, max_frames=24, max_width=192):
    """Read a bounded initial clip, preserving source FPS. Requires PyAV."""
    import av
    from PIL import Image
    if max_frames<1 or max_width<1: raise ValueError('Clip limits must be positive')
    frames=[]
    try:
        with av.open(io.BytesIO(data)) as container:
            stream=next((s for s in container.streams if s.type=='video'),None)
            if stream is None: raise ValueError('The file contains no video stream')
            fps=float(stream.average_rate) if stream.average_rate else 24.0
            for frame in container.decode(stream):
                image=frame.to_image().convert('RGB')
                if image.width>max_width:
                    image=image.resize((max_width,max(1,round(image.height*max_width/image.width))),Image.Resampling.LANCZOS)
                frames.append(np.asarray(image))
                if len(frames)>=max_frames: break
    except av.error.FFmpegError as exc:
        raise ValueError('Unable to decode this video. Try an MP4 with H.264 video.') from exc
    if not frames: raise ValueError('No decodable video frames found')
    return np.stack(frames),fps
