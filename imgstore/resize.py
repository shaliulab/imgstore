import json

import cv2
import numpy as np

from .stores import new_for_filename, new_for_format, _ImgStore


class ImgStoreResizer:
    def __init__(self, store):
        if isinstance(store, str):
            self._store = new_for_filename(store_path)
        elif isinstance(store, _ImgStore):
            self._store = store

        self._dest = None

    def resize(self, time_range, dest, **user_kwargs):

        kwargs = {
            "imgshape": self._store._metadata["imgshape"],
            "chunksize": self._store._metadata["chunksize"],
            "framerate": self._store._metadata["framerate"],
        }

        kwargs.update(user_kwargs)

        self._dest = new_for_format(fmt="mjpeg/avi", path=dest, **kwargs)

        img, (frame_number_0, timestamp_0) = self._store._get_image_by_time(
            time_range[0]
        )

        timestamp = timestamp_0

        self._dest.add_image(img, 0, 0)
        while timestamp < time_range[-1]:
            img, (frame_number, timestamp) = self._store.get_next_image()
            frame_number_dest = frame_number - frame_number_0
            timestamp_dest = timestamp - timestamp_0
            self._dest.add_image(img, frame_number_dest, timestamp_dest)

        self._dest.close()
