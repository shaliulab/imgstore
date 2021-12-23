import unittest
from imgstore.tests import TEST_DATA_DIR
import os.path

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

STORE_PATH = os.path.join(TEST_DATA_DIR, "imgstore_1", "metadata.yaml")


class TestCompat(unittest.TestCase):

    QUERY_FRAME_COUNT = 100

    def setUp(self):
        store = new_for_filename(STORE_PATH)
        _ = store.get_image(self.QUERY_FRAME_COUNT)
        self._store = store

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

        target_width = self._store._metadata["imgshape"][1]
        target_height = self._store._metadata["imgshape"][0]

        self.assertEqual(width, target_width)
        self.assertEqual(height, target_height)

    def test_get_pos(self):
        """
        Test that the position of the videos can be fetched
        """

        pos_msec = self._store.get(CAP_PROP_POS_MSEC)
        pos_frames = self._store.get(CAP_PROP_POS_FRAMES)

        # next_frame_number = self._store.get_frame_metadata()["frame_number"][self.QUERY_FRAME_COUNT + 1]
        next_frame_number = self._store._chunk_current_frame_idx + 1

        next_timestamp = self._store._get_chunk_metadata(self._store._chunk_n)[
            "frame_time"
        ][self._store._chunk_current_frame_idx + 1]
        t0 = self._store._get_chunk_metadata(self._store._chunk_n)[
            "frame_time"
        ][0]
        next_timestamp -= t0

        self.assertEqual(pos_msec, next_timestamp)
        self.assertEqual(pos_frames, next_frame_number)

    def test_get_framecount(self):

        frame_count = self._store.get(CAP_PROP_FRAME_COUNT)
        self.assertEqual(
            frame_count,
            len(
                self._store._get_chunk_metadata(self._store._chunk_n)[
                    "frame_number"
                ]
            ),
        )

        pos_rel = self._store.get(CAP_PROP_POS_AVI_RATIO)
        self.assertEqual(
            pos_rel, (self._store._chunk_current_frame_idx + 1) / frame_count
        )

    def test_get_fps(self):
        fps = self._store.get(CAP_PROP_FPS)

        framerate = self._store._metadata["framerate"]
        self._store._load_chunk(0)
        video_fps = self._store._cap.get(CAP_PROP_FPS)
        self.assertEqual(fps, framerate)
        self.assertEqual(fps, video_fps)

    def test_release(self):

        self._store.release()
        # test that the store is closed


if __name__ == "__main__":
    unittest.main()
