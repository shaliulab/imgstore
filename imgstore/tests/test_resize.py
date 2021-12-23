import unittest
import os.path
import shutil

from imgstore import new_for_filename
from imgstore.resize import ImgStoreResizer
from imgstore.tests import TEST_DATA_DIR
import cv2
from cv2 import CAP_PROP_FRAME_COUNT

TEST_STORE_PATH = os.path.join(TEST_DATA_DIR, "imgstore_1", "metadata.yaml")
DEST_STORE_PATH = os.path.join(TEST_DATA_DIR, "test_store")

class TestResize(unittest.TestCase):

    def setUp(self):
        self._src = new_for_filename(TEST_STORE_PATH)

    def tearDown(self):
        self._src.close()
        shutil.rmtree(DEST_STORE_PATH)


    def test_resize_can_merge(self):
        resizer = ImgStoreResizer(self._src)
        resizer.resize([1000, 10000], DEST_STORE_PATH, chunksize = 4500)

        avi_path = os.path.join(
            DEST_STORE_PATH,
            "000000.avi"
        )

        self.assertTrue(
            os.path.exists(
                avi_path
            )
        )

        self._src._load_chunk(0)
        cap = self._src._cap
        frame_count = cap.get(CAP_PROP_FRAME_COUNT)
        self.assertEqual(frame_count, len(self._src._chunk_md["frame_number"]))

    def test_resize_can_split(self):
        resizer = ImgStoreResizer(self._src)
        resizer.resize([1000, 2000], DEST_STORE_PATH, chunksize = 5)
        avi_path = os.path.join(
            DEST_STORE_PATH,
            "000000.avi"
        )

        self.assertTrue(
            os.path.exists(
                avi_path
            )
        )

        cap = cv2.VideoCapture(avi_path)
        frame_count = cap.get(CAP_PROP_FRAME_COUNT)
        self.assertEqual(frame_count, 5)





if __name__ == "__main__":
    unittest.main()
