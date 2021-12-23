import unittest
import logging
import imgstore
import numpy as np


STORE_PATH = "./tests/static_data/corrupt_store/metadata.yaml"


class TestOpenCorruptStore(unittest.TestCase):
    def test_chunk_numbers_skips_corrupt_chunk(self):

        with self.assertLogs(level="WARNING"):
            store = imgstore.new_for_filename(STORE_PATH)

        img, (fn, t) = store.get_next_image()
        self.assertEqual(fn, 4500)
        self.assertIsInstance(img, np.ndarray)
