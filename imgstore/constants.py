from cv2 import VideoWriter_fourcc as FourCC

try:
    # python 3
    # noinspection PyProtectedMember
    from subprocess import DEVNULL

    # noinspection PyShadowingBuiltins
    xrange = range
except ImportError:
    # python 2
    import os

    DEVNULL = open(os.devnull, "r+b")

STORE_MD_KEY = "__store"
STORE_MD_FILENAME = "metadata.yaml"
STORE_LOCK_FILENAME = ".lock"
STORE_INDEX_FILENAME = ".index.sqlite"

EXTRA_DATA_FILE_EXTENSIONS = (
    ".extra.json",
    ".extra_data.json",
    ".extra_data.h5",
)

FRAME_MD = ("frame_number", "frame_time", "frame_in_chunk")
ENCODER_FORMAT_GPU="h264_nvenc/mp4" # CUDA
ENCODER_FORMAT_CPU="mjpeg/avi" # NO CUDA

FMT_TO_CODEC = {
        "mjpeg": FourCC("M", "J", "P", "G"),
        "h264/avi": "libx264",
        "mjpeg/avi": FourCC("M", "J", "P", "G"),
        "mjpeg/mp4": FourCC("M", "J", "P", "G"),
        "h264/mkv": FourCC("H", "2", "6", "4"),
        "avc1/mp4": FourCC("a", "v", "c", "1"),
        "h264/mp4": FourCC("H", "2", "6", "4"),
        "x264/mp4": FourCC("X", "2", "6", "4"),
        "h264_nvenc/mp4": "h264_nvenc",
    }