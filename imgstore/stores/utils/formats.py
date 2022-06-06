import numpy as np
import os.path
import logging
import tempfile
import cv2
from imgstore.configuration import load_config, save_config

logger = logging.getLogger(__name__)

from imgstore.util import FourCC
try:
    import cv2cuda # type: ignore
    ENCODER_FORMAT_GPU="h264_nvenc/mp4" # CUDA
    ENCODER_FORMAT_CPU="mjpeg/avi" # NO CUDA

except ImportError:
    cv2cuda = None

# https://answers.opencv.org/question/100967/codecs-list/

RAINBOW = [
  (148, 0, 211),
  (75, 0, 130),
  (0, 0, 255),
  (0, 255, 0),
  (255, 255, 0),
  (255, 127, 0),
  (255, 0 , 0),
]

def get_colorbar(shape):
    height, width = shape

    bar_width = width // len(RAINBOW)
    offset = width - bar_width * len(RAINBOW)

    bar_height = height

    bars=[]
    for color in RAINBOW:
        bars.append(
            np.ones((bar_height, bar_width, 3), np.uint8) * np.array(color, np.uint8).reshape(1,1,3)
        )
    if offset != 0:
        bars.append(
            np.ones((bar_height, offset, 3), np.uint8) * np.array(RAINBOW[-1], np.uint8).reshape(1,1,3)
        )

    colorbar=np.hstack(bars)
    
    return colorbar

def verify_fourcc(format, fourcc, cache=True):

    config = load_config()
    if cache:
        if format in config["codecs"]:
            return config["codecs"][format]
        
    extension = format.split("/")[1]
    dest_file = tempfile.NamedTemporaryFile(suffix="."+extension).name
    fps = 30
    frameSize=(768, 1024)
    frameShape=frameSize[::-1]

    video_writer = cv2.VideoWriter(
        dest_file,
        fourcc,
        fps=fps,
        frameSize=frameSize,
        isColor=False,
    )

    for _ in range(fps*2):
        frame = get_colorbar(frameShape)
        video_writer.write(frame)
    video_writer.release()

    available=os.path.isfile(dest_file)

    config["codecs"][format] = available
    save_config(config)
    return available



def verify_all(all_fourcc, **kwargs):

    available = {}

    for format, fourcc in all_fourcc.items():
        if verify_fourcc(format, fourcc, **kwargs):
            available[format] = fourcc

    return available

def load_and_verify_opencv_formats(**kwargs):

    available=[]

    opencv_formats = {
        'mjpeg/avi': "MJPG",
        'h264/mkv': "H264",
        'avc1/mp4': "avc1",
    }

    opencv_fourcc = {
        k: FourCC(*v) for k, v in opencv_formats.items()
    }

    available = verify_all(opencv_fourcc, **kwargs)
    return available

def load_and_verify_cv2cuda_formats(**kwargs):

    if cv2cuda:
        cv2cuda_fourcc = {
            ENCODER_FORMAT_GPU: "h264_nvenc"
        }
    else:
        cv2cuda_fourcc = {}


    # available = verify_all(cv2cuda_fourcc)
    available = cv2cuda_fourcc
    return available


def get_formats(**kwargs):

    formats = {}

    opencv_fourcc = load_and_verify_opencv_formats(**kwargs)
    cv2cuda_fourcc = load_and_verify_cv2cuda_formats(**kwargs)
    formats.update(opencv_fourcc)
    formats.update(cv2cuda_fourcc)
    logger.debug(f"Available formats: {formats}")
    return formats


__all__ = ["get_formats"]