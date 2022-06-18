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

try:
    import bloscpack # type: ignore
except Exception:
    bloscpack=None

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

opencv_formats = {
    # 'mjpeg/avi': "MJPG",
    # 'h264/mkv': "H264",
    # 'h264/avi': "H264",
    # 'avc1/mp4': "avc1",
    # 'mp4v/mp4': "mp4v",
    # 'divx/avi': "DIVX",
    # 'divx/avi': "divx",
    'avc1/avi': "avc1",
    # 'mpeg/avi': "MPEG",    
}

cv2_fmts = {'tif', 'png', 'jpg', 'ppm', 'pgm', 'bmp'}
raw_fmts = {'npy', 'bpk'}
frameSize=(768, 1024)
frameShape=frameSize[::-1]

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


    if fourcc == "h264_nvenc":
        if cv2cuda is not None:
            video_writer_f=cv2cuda.VideoWriter
            apiPreference="FFMPEG"
        else:
            fourcc = FourCC(*opencv_formats['avc1/mp4'])
            video_writer_f=cv2.VideoWriter
            apiPreference=cv2.CAP_FFMPEG
    else:
        video_writer_f=cv2.VideoWriter
        apiPreference=cv2.CAP_FFMPEG

    
    video_writer = video_writer_f(
        dest_file,
        apiPreference=apiPreference,
        fourcc=fourcc,
        fps=fps,
        frameSize=frameSize,
        isColor=False,
    )


    for _ in range(fps*10):
        frame = get_colorbar(frameShape)
        video_writer.write(frame)

    
    video_writer.release()

    cap=cv2.VideoCapture(dest_file)
    ret, frame=cap.read()
    available= os.path.isfile(dest_file) and \
        ret and frame is not None
    

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

    opencv_fourcc = {}
    for k, v in opencv_formats.items():
        opencv_fourcc[k] =FourCC(*v)

    available = verify_all(opencv_fourcc, **kwargs)
    return available

def load_and_verify_cv2cuda_formats(**kwargs):

    if cv2cuda is not None:
        cv2cuda_fourcc = {
            ENCODER_FORMAT_GPU: "h264_nvenc"
        }
    else:
        cv2cuda_fourcc = {}


    available = verify_all(cv2cuda_fourcc)
    available = cv2cuda_fourcc
    return available


def verify_dirf(format, fourcc, cache=True):
    
    config = load_config()
    if cache:
        if format in config["codecs"]:
            return config["codecs"][format]
        
    extension=format
    dest_file = tempfile.NamedTemporaryFile(suffix="."+extension).name
    img=get_colorbar(frameShape)
    try:
        if extension == "npy":
            np.save(dest_file, img)
        elif bloscpack is not None and extension == "bpk":
            bloscpack.pack_ndarray_file(img, dest_file)
        else:
            cv2.imwrite(dest_file, img)
        
    except Exception as error:
        pass
    available=os.path.isfile(dest_file)
    config["codecs"][format] = available
    save_config(config)
    return available

def load_and_verify_dir_formats(**kwargs):

    available={}

    for fmt in cv2_fmts:
        available[fmt]=verify_dirf(fmt, fmt, **kwargs)

    return available

def load_and_verify_raw_formats(**kwargs):

    available={}
    for fmt in raw_fmts:
        available[fmt]=verify_dirf(fmt, fmt, **kwargs)

    return available

def get_formats(video=False, directory=False, raw=False, **kwargs):

    opencv_fourcc = load_and_verify_opencv_formats(**kwargs)
    cv2cuda_fourcc = load_and_verify_cv2cuda_formats(**kwargs)
    dir_formats = load_and_verify_dir_formats(**kwargs)
    raw_formats = load_and_verify_raw_formats(**kwargs)

    video_formats={}
    video_formats.update(opencv_fourcc)
    video_formats.update(cv2cuda_fourcc)

    formats={}
    formats.update(video_formats)
    formats.update(dir_formats)
    formats.update(raw_formats)

    logger.debug(f"Available formats: {formats}")

    formats={}
    if video:
        formats.update(video_formats)
    if directory:
        formats.update(dir_formats)
    if raw:
        formats.update(raw_formats)
    
    return formats

__all__ = ["get_formats"]