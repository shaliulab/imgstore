import os.path
import logging


import numpy as np
import cv2
from cv2 import (
    CAP_PROP_FRAME_WIDTH,
    CAP_PROP_FRAME_HEIGHT,
    CAP_PROP_POS_MSEC,
    CAP_PROP_FPS,
    CAP_PROP_POS_FRAMES,
)

from .stores import new_for_filename, _extract_store_metadata

logger = logging.getLogger(__name__)

class MultiStore:

    _LAYOUT = (2, 1)

    @classmethod
    def new_for_filename(cls, store, *args, **kwargs):

        store_list = [store]

        metadata = _extract_store_metadata(store)
        if "extra_cameras" in metadata:
            for extension in metadata["extra_cameras"]:
                extension_full_path = os.path.join(
                    os.path.dirname(store),
                    extension
                )
                store_list.append(extension_full_path)

        delta_time = metadata.get("delta_time", None)
        return cls(store_list, *args, delta_time=delta_time, **kwargs)


    def __init__(self, store_list, ref_chunk, chunk_numbers=None, delta_time=None, layout=None, adjust_by="resize", **kwargs):


        main_store = store_list[0]

        self._stores = [new_for_filename(
            main_store,
            chunk_numbers=chunk_numbers,
            **kwargs
        )]

        ref_chunk = int(ref_chunk)
        self._main_store = self._stores[0]
        ref_chunk = int(ref_chunk)
        self._main_store._chunk = ref_chunk
        self._main_store._load_chunk(ref_chunk)
        self._main_store.reset_to_first_frame()

        self._stores.extend([
            new_for_filename(store_path, **kwargs)
            for store_path in store_list[1:]
        ])

        self._store_list = sorted(
            [store for store in self._stores],
            key=lambda x: x._metadata["framerate"]
        )

        # sync
        for store in self._stores[1:]:
            store._set_posmsec(self._stores[0].frame_time, absolute=True)

        if layout is None:
            self._layout = self._LAYOUT

        self._adjust_by = adjust_by

        self._width = None
        self._height = None

        self._delta_time = delta_time
        self._delta_time_generator = sorted(
            [
                store for store in self._store_list
            ], key=lambda x: x._metadata["framerate"]
        )[-1]



    def _apply_layout(self, imgs):
        # place the imgs by row
        # so the first row is filled,
        # then second, until last

        ncols = self._layout[1]
        nrows = self._layout[0]
        i = 0
        i_last = ncols
        rows = []

        for row_i in range(nrows):
            rows.append(self.make_row(imgs[i:i_last]))
            i = i_last
            i_last = i + ncols

        rows = sorted(rows, key=lambda x: -x.shape[1])
        width = rows[0].shape[1]

        for i in range(1, len(rows)):
            rows[i] = self._adjust_width(rows[i], width)

        return np.vstack(rows)

    def _adjust_width(self, img, width):
        width_diff = int(width - img.shape[1])
        if width_diff != 0:
            if self._adjust_by == "pad":
                img = self._pad_img(img, width_diff)

            elif self._adjust_by == "resize":
                img = self._resize_img(img, width_diff)

        assert img.shape[1] == width

        return img

    @staticmethod
    def _resize_img(img, width_diff):

        ratio = (img.shape[1] + width_diff) / img.shape[1]

        img = cv2.resize(
            img, (img.shape[1] + width_diff, int(ratio * img.shape[0])), cv2.INTER_AREA
        )
        return img


    @staticmethod
    def _pad_img(img, width_diff):
        if width_diff % 2 == 0:
            odd_pixel = 0
        else:
            odd_pixel = 1

        img = cv2.copyMakeBorder(
            img,
            0,
            0,
            width_diff // 2,
            width_diff // 2 + odd_pixel,
            cv2.BORDER_CONSTANT,
            value=0,
        )
        return img


    @staticmethod
    def make_row(imgs):
        height = imgs[0].shape[0]
        for img in imgs[1:]:
            assert img.shape[0] == height

        return np.hstack(imgs)

    def _read(self):
        imgs = []

        if self._delta_time is None:
            import ipdb; ipdb.set_trace()
            img, (frame_number, frame_time) = self._delta_time_generator.get_next_image()
            imgs.append(img)
            logger.warning(f"Reading frame #{frame_number} at time {frame_time} from {self._delta_time_generator}")
            for store in self._store_list:
                if store is self._delta_time_generator:
                    continue
                else:
                    if frame_time >= min(store._chunk_md["frame_time"]):
                        img, (frame_number_, frame_time_) = store._get_image_by_time(frame_time, direction="past")
                    else:
                        logger.warning("Cannot find a frame further back in the past")
                        frame_number = min(store._chunk_md["frame_number"])
                        img, (frame_number_, frame_time_)= store.get_image(frame_number)

                    logger.warning(f"Reading frame #{frame_number_} at time {frame_time_} from {store}")
                    imgs.append(img)


        else:
            for store in self._store_list:
                img, (frame_number_, frame_timestamp_)= store._get_image_by_time(
                    store.frame_time + self._delta_time
                )
                imgs.append(img)

        ret = len(imgs) == len(self._store_list)
        return ret, imgs

    def read(self):
        ret, imgs = self._read()
        if ret:
            img = self._apply_layout(imgs)
            return ret, img

        else:
            return False, None

    def release(self):
        for store in self._store_list:
            store.close()

    def close(self):
        self.release()


    def _read_test_frame(self):
        pos_msec = self.get(CAP_PROP_POS_MSEC)
        ret, frame = self.read()
        self.set(CAP_PROP_POS_MSEC, pos_msec)
        return frame


    def get(self, index):

        # TODO Dont hardcode the layout here
        if index == CAP_PROP_FRAME_WIDTH:
            width = self._read_test_frame().shape[1]
            return width

        elif index == CAP_PROP_FRAME_HEIGHT:
            height = self._read_test_frame().shape[0]
            return height

        elif index == CAP_PROP_FPS:
            fps = self._store_list[-1]._metadata["framerate"]
            return fps

        else:
            return self._store_list[0].get(index)

    def set(self, index, value):


        if index in [CAP_PROP_POS_FRAMES, CAP_PROP_POS_MSEC]:
            try:
                self._delta_time_generator.set(index, value, absolute=True)
            except:
                import ipdb; ipdb.set_trace()
            for store in self._store_list:
                if store is self._delta_time_generator:
                    continue
            else:
                store._set_posmsec(
                    self._delta_time_generator.frame_time,
                    absolute=True,
                    direction="past"
                )

        else:

            for store in self._store_list:
                store.set(index, value)

    def get_image(self, frame_number):
        raise Exception("get_image method is not meaningful in a multistore")

    def get_image_by_time(self, timestamp):

        imgs = []
        for store in self._store_list:
            img, _ = store._get_image_by_time(timestamp)
            imgs.append(img)

        return imgs

    @property
    def frame_time(self):
        return self._store_list[-1].frame_time

    def __getattr__(self, attr):
        return getattr(self._store_list[0], attr)

    @property
    def data_interval(self):
        _, (frame_number, frame_time) = self._delta_time_generator.get_next_image()

        main_store_interval = self._main_store.get_data_interval(what="frame_time", pad=10)
        self._delta_time_generator._set_posmsec(main_store_interval[0], absolute=True)
        begin = self._delta_time_generator.frame_number
        self._delta_time_generator._set_posmsec(main_store_interval[1], absolute=True)
        end = self._delta_time_generator.frame_number
        self._delta_time_generator.get_image(frame_number - 1)
        

        assert begin < end
        return (begin, end)

