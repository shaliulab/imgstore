import unittest
import os.path
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

    def setUp(self):

        self._multistore = MultiStore([
            HR_STORE_PATH,
            LR_STORE_PATH,
        ])

    def test_layout(self):
        
        ret, frame = self._multistore.read()
        height, width = frame.shape
        self.assertEqual(height, 1974)
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
            1974
        )

    
    def tearDown(self):
        self._multistore.close()


if __name__ == "__main__":
    unittest.main()
