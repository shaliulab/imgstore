import numpy as np
import os.path
import logging
import tempfile
import cv2
from imgstore.configuration import load_config, save_config
from imgstore.stores.utils.verify import get_colorbar, verify_fourcc, frameSize, frameShape
from imgstore.stores.utils.formats_constants import *

logger = logging.getLogger(__name__)

from imgstore.util import FourCC
try:
    import cv2cuda # type: ignore
    ENCODER_FORMAT_GPU="h264_nvenc/mp4" # CUDA
    ENCODER_FORMAT_CPU="mjpeg/avi" # NO CUDA

except ImportError:
    logger.info("cv2cuda is not available")
    cv2cuda = None

try:
    import bloscpack # type: ignore
except Exception:
    bloscpack=None

# https://answers.opencv.org/question/100967/codecs-list/



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
    try:
        cv2cuda_fourcc = load_and_verify_cv2cuda_formats(**kwargs)
    except:
        print(f"cv2cuda may be installed but the h264_nvenc codec cannot be initialized")
        cv2cuda_fourcc = {}

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
