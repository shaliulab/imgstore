import unittest
unittest.TestLoader.sortTestMethodsUsing = None
import os.path
import numpy as np
import cv2
from cv2 import (
    CAP_PROP_FPS,
    CAP_PROP_FRAME_WIDTH,
    CAP_PROP_FRAME_HEIGHT,
)

from imgstore.tests import TEST_DATA_DIR
from imgstore.multistores import MultiStore

STORE_PATH = os.path.join(TEST_DATA_DIR, "imgstore_1", "metadata.yaml")

LR_STORE_FPS = 70

class TestMultiStore(unittest.TestCase):

    _adjust_by="pad"
    _target_height = 1974


    def setUp(self):

        self._multistore = MultiStore.new_for_filename(
            STORE_PATH,
            adjust_by=self._adjust_by
        )


    def test_get_fps(self):
        fps = self._multistore.get(CAP_PROP_FPS)
        self.assertEqual(fps, 70.0)


    def test_read(self):
        ret, imgs = self._multistore._read()

        self.assertTrue(ret)
        self.assertTrue(isinstance(imgs, list))

    def test_layout(self):

        imgs = [
            np.random.randint(0, 255, (100, 200), dtype=np.uint8),
            np.random.randint(0, 255, (100, 100), dtype=np.uint8),
        ]

        frame = self._multistore._apply_layout(imgs)
        height, width = frame.shape

        if self._adjust_by == "pad":
            self.assertEqual(height, 200)
        elif self._adjust_by == "resize":
            self.assertEqual(height, 300)

        self.assertEqual(width, 200)

    
    def test_getters(self):
        self.assertEqual(
            self._multistore.get(CAP_PROP_FPS),
            LR_STORE_FPS
        )

        self.assertEqual(
            self._multistore.get(CAP_PROP_FRAME_WIDTH),
            1514
        )


        self.assertEqual(
            self._multistore.get(CAP_PROP_FRAME_HEIGHT),
            self._target_height
        )


    def test_frame(self):
        ret, frame = self._multistore.read()
        dest = os.path.join(
            TEST_DATA_DIR,
            "multistore_frame.png"
        )
        print(dest)

        cv2.imwrite(dest, frame)

    
    def tearDown(self):
        self._multistore.close()


class TestMultiStoreResize(TestMultiStore):
    _adjust_by="resize"
    _target_height = 3031


if __name__ == "__main__":
    unittest.main()
