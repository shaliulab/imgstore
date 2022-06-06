import os.path
from imgstore.stores import multi as multistores
from imgstore.tests import TEST_DATA_DIR

MULTISTORE_PATH = os.path.join(TEST_DATA_DIR, "multistore")


def test_get_image(store):
    img, (frame_number, frame_time) = store.get_image(1000)

def test_get_nearest_image(store):

    img, (frame_number, frame_time) = store.get_nearest_image(450)

def test_get_next_image(store):
    img, (frame_number, frame_time) = store.get_next_image()
    

def test_multistore():

    with multistores.new_for_filename(MULTISTORE_PATH) as multistore:
        test_get_image(multistore)
        test_get_nearest_image(multistore)
        test_get_next_image(multistore)



test_multistore()