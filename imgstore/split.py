import json

import cv2
import numpy as np

from .stores import new_for_filename, new_for_format

class ImgStoreSplitter:

    def __init__(self, store_path):
        self._store = new_for_filename(store_path)

    def split(self, time_range, dest, **kwargs):


        import ipdb; ipdb.set_trace()
        kwargs.update({
            "imgshape": self._store._metadata["imgshape"],
            "chunksize": self._store._metadata["chunksize"],
            "framerate": self._store._metadata["framerate"]
        })

        dest_store = new_for_format(fmt="mjpeg/avi", path=dest, **kwargs)

        img, (frame_number_0, timestamp_0) = self._store._get_image_by_time(time_range[0])

        timestamp = timestamp_0

        dest_store.add_image(img, 0, 0)
        while timestamp < time_range[-1]:
            img, (frame_number, timestamp) = self._store._get_next_image()
            frame_number_dest = frame_number - frame_number_0
            timestamp_dest = timestamp - timestamp_0
            dest_store.add_image(img, frame_number_dest, timestamp_dest)

        dest_store.close()

