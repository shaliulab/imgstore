import unittest
import os.path
import cv2
from cv2 import (
    CAP_PROP_FPS,
    CAP_PROP_FRAME_WIDTH,
    CAP_PROP_FRAME_HEIGHT,
)

from imgstore.tests import TEST_DATA_DIR
from imgstore.multistores import MultiStore

HR_STORE_PATH = os.path.join(TEST_DATA_DIR, "imgstore_1", "metadata.yaml")
LR_STORE_PATH = os.path.join(TEST_DATA_DIR, "imgstore_1", "lowres", "metadata.yaml")
LR_STORE_FPS = 70

class TestMultiStore(unittest.TestCase):

    _adjust_by="pad"
    _target_height = 1974


    def setUp(self):

        self._multistore = MultiStore(
            store_list = [
            HR_STORE_PATH,
            LR_STORE_PATH,
        ],
            adjust_by=self._adjust_by
        )

    def test_layout(self):
        
        ret, frame = self._multistore.read()
        self.assertTrue(ret)
        height, width = frame.shape

        self.assertEqual(height, self._target_height)
        self.assertEqual(width, 1514)

    
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
