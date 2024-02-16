import os.path
import cv2
import numpy as np
import tempfile
from imgstore.configuration import load_config, save_config
from imgstore.stores.utils.formats_constants import opencv_formats
from imgstore.util import FourCC

frameSize=(768, 1024)
frameShape=frameSize[::-1]
try:
    import cv2cuda
except:
    cv2cuda=None


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


    if fourcc == "h264_nvenc":
        if cv2cuda is not None:
            video_writer_f=cv2cuda.VideoWriter
            apiPreference="FFMPEG"
        else:
            fourcc = FourCC(*opencv_formats['mp4v/mp4'])
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
        frame = get_colorbar(frameShape)[:,:,0]
        video_writer.write(frame)


    video_writer.release()

    cap=cv2.VideoCapture(dest_file)
    ret, frame=cap.read()
    available= os.path.isfile(dest_file) and \
        ret and frame is not None


    config["codecs"][format] = available
    save_config(config)
    return available

