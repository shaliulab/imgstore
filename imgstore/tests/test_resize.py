import unittest
import os.path
from imgstore import new_for_filename
from imgstore.resize import ImgStoreResizer
from cv2 import CAP_PROP_FRAME_COUNT

TEST_STORE_PATH = "./tests/static_data/imgstore_1"
DEST_STORE_PATH = "./tests/temp_data/test_store"

class TestResize(unittest.TestCase):

    def setUp(self):
        self._src = new_for_filename(TEST_STORE_PATH)
    
    def tearDown(self):
        self._src.close()

    
    def test_resize_can_merge(self):
        resizer = ImgStoreResizer(self._src)
        resizer.resize([1000, 10000], DEST_STORE_PATH, chunksize = 4500)
        
        avi_path = os.path.join(
            os.path.dirname(DEST_STORE_PATH),
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


if __name__ == "__main__":
    unittest.main()
