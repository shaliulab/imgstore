import os.path
import imgstore.stores.multi as imgstore
from imgstore.tests import TEST_DATA_DIR
from imgstore.constants import STORE_MD_FILENAME

MULTISTORE_PATH = os.path.join(TEST_DATA_DIR, "multistore", STORE_MD_FILENAME)


def test_get_image(store):
    # TODO
    img, (frame_number, frame_time) = store.get_image(100)

def test_get_nearest_image(store):
    # TODO
    img, (frame_number, frame_time) = store.get_nearest_image(45)

def test_get_next_image(store):
    # TODO
    img, (frame_number, frame_time) = store.get_next_image()
    

def test_multistore():

    with imgstore.new_for_filename(MULTISTORE_PATH) as multistore:
        test_get_image(multistore)
        test_get_nearest_image(multistore)
        test_get_next_image(multistore)



test_multistore()
