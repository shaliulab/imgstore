import os.path
import unittest
import time
import psutil
import cv2cuda
import numpy as np
import imgstore
from imgstore.constants import ENCODER_FORMAT_GPU

DATA_DIR = "./imgstore/tests/saved_store/"
VIDEO_PATH = os.path.join(
    DATA_DIR,
    "cv2cuda-test.mp4"
)

IMGSHAPE=(200, 200)
FPS=100
CHUNK_DURATION=5
N_FRAMES=2000
FORMAT=ENCODER_FORMAT_GPU
start_time = time.time()

def get_frame(shape):
    frame = np.random.randint(0, 255, shape)
    return frame


def get_frame_timestamp(*args, **kwargs):
    frame = get_frame(*args, **kwargs)
    timestamp = time.time() - start_time
    return frame, timestamp


class TestCV2Cuda(unittest.TestCase):

    def test_store_cuda(self):

        chunksize = FPS * CHUNK_DURATION

        store = imgstore.new_for_format(
            mode="w",
            fmt=FORMAT,
            framerate=FPS,
            basedir=VIDEO_PATH,
            imgshape=IMGSHAPE,
            chunksize=chunksize,
            imgdtype=np.uint8
        )

        self.assertEqual(
            sum(["ffmpeg" in e.name() for e in psutil.process_iter()]),
            1
        )

        for i in range(N_FRAMES):
            frame, timestamp = get_frame_timestamp(IMGSHAPE)
            store.add_image(frame, i, timestamp)
        
        store.close()

        self.assertEqual(
            sum(["ffmpeg" in e.name() for e in psutil.process_iter()]),
            0
        )


if __name__ == "__main__":
    unittest.main()