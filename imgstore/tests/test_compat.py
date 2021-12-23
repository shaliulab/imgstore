import unittest

import numpy as np
from imgstore import new_for_filename

from cv2 import (
    CAP_PROP_POS_MSEC,
    CAP_PROP_POS_FRAMES,
    CAP_PROP_POS_AVI_RATIO,
    CAP_PROP_FRAME_WIDTH,
    CAP_PROP_FRAME_HEIGHT,
    CAP_PROP_FPS,
    CAP_PROP_FOURCC,
    CAP_PROP_FRAME_COUNT,
)    

STORE_PATH = "tests/static_data/store/metadata.yaml"

class TestCompat(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        store = new_for_filename(STORE_PATH)
        _ = store.get_image(1000)
        cls._store = store

    def test_read(self):
        ret, img = self._store.read()

        self.assertTrue(ret)
        self.assertTrue(isinstance(img, np.ndarray))
        # Test store timestamp or frame number


    def test_get_resolution(self):
        """
        Test that width and height can be fetched
        """

        width = self._store.get(CAP_PROP_FRAME_WIDTH)
        height = self._store.get(CAP_PROP_FRAME_HEIGHT)

        self.assertEqual(width, self._WIDTH)
        self.assertEqual(height, self._HEIGHT)

    
    def test_get_pos(self):
        """
        Test that the position of the videos can be fetched
        """

        pos_msec = self._store.get(CAP_PROP_POS_MSEC)
        pos_frames = self._store.get(CAP_PROP_POS_FRAMES)
        pos_rel = self._store.get(CAP_PROP_POS_AVI_RATIO)

        self.assertEqual(pos_msec, )
        self.assertEqual(pos_frames, )
        self.assertEqual(pos_rel, )
    
    def test_get_framecount(self):

        frame_count = self.store.get(CAP_PROP_FRAME_COUNT)
        self.assertEqual(frame_count, )

    
    def test_get_fps(self):
        fps = self._store.get(CAP_PROP_FPS)
        self.assertEqual(fps, )


    def test_release(self):
        
        self._store.release()
        # test that the store is closed

    
if __name__ == "__main__":
    unittest.main()

