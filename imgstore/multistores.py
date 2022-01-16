import os.path
import logging
import traceback
import glob
import datetime
import pandas as pd
import numpy as np
import cv2
from cv2 import (
    CAP_PROP_FRAME_WIDTH,
    CAP_PROP_FRAME_HEIGHT,
    CAP_PROP_POS_MSEC,
    CAP_PROP_FPS,
    CAP_PROP_POS_FRAMES,
    CAP_PROP_FRAME_COUNT,
)

from .stores import new_for_filename, _extract_store_metadata

logger = logging.getLogger(__name__)

class MultiStore:

    _TOLERANCE = 1000

    @classmethod
    def new_for_filename(cls, store, chunk_numbers=None, *args, **kwargs):

        store_list = [store]

        metadata = _extract_store_metadata(store)
        if "extra_cameras" in metadata:
            for extension in metadata["extra_cameras"]:
                extension_full_path = os.path.join(
                    os.path.dirname(store),
                    extension
                )
                store_list.append(extension_full_path)

        if chunk_numbers is None:
            index = sorted(glob.glob(
                os.path.join(
                    os.path.dirname(store),
                    "*.npz"
                )
            ))

            chunk_numbers = []
            for file in index:
                chunk, ext = os.path.splitext(os.path.basename(file))
                chunk = int(chunk)
                chunk_numbers.append(chunk)

        if "corrupt_chunks" in metadata:
            for corrupt_chunk in metadata["corrupt_chunks"]:
                if corrupt_chunk in chunk_numbers:
                    chunk_numbers.pop(chunk_numbers.index(corrupt_chunk))
                    logger.warning(f"Chunk {corrupt_chunk} is corrupt. Omitting")

        delta_time = metadata.get("delta_time", None)
        return cls(store_list, *args, delta_time=delta_time, chunk_numbers=chunk_numbers, **kwargs)

    @property
    def width(self):
        if self._width is None:
            self._width = self.get(CAP_PROP_FRAME_WIDTH)

        return self._width

    @property
    def height(self):
        if self._height is None:
            self._height = self.get(CAP_PROP_FRAME_HEIGHT)

        return self._height

    @staticmethod
    def log_position_along(store):
        store_frame_time_human=datetime.datetime.fromtimestamp(store.frame_time/1000).strftime('%H:%M:%S.%f')
        print(
            f"store {store} is set to\n"
            f"* chunk {store._chunk_n}\n"
            f"* frame_in_chunk {store._chunk_current_frame_idx}\n"
            f"* frame_number {store.frame_number}\n"
            f"* frame_time {store.frame_time} ({store_frame_time_human})\n"
        )

    @property
    def delta_frame_number(self):
        return self._delta_frame_number


    def __init__(self, store_list, ref_chunk, chunk_numbers=None, delta_time=None, layout=None, adjust_by="resize", **kwargs):


        main_store = store_list[0]

        self._stores = [new_for_filename(
            main_store,
            chunk_numbers=chunk_numbers,
            **kwargs
        )]

        self._data_interval = None

        self._width = None
        self._height = None
        self._crossindex = None
        self._delta_frame_number = 0


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

        for store in self._stores:
            logger.info(f"Computing chunk index of {store}")
            _ = store._index.chunk_index

        self._store_list = sorted(
            [store for store in self._stores],
            key=lambda x: x._metadata["framerate"]
        )

        # sync
        for store in self._stores[1:]:
            store._set_posmsec(self._stores[0].frame_time, absolute=True)

        self._layout = layout
        self._main_only = False


        self._adjust_by = adjust_by

        self._width = None
        self._height = None

        self._delta_time = delta_time
        self._delta_time_generator = sorted(
            [
                store for store in self._store_list
            ], key=lambda x: x._metadata["framerate"]
        )[-1]
        self.get_crossindex()


    @property
    def layout(self):
        if self._layout is None:
            return (len(self._stores), 1)
        else:
            return self._layout



    def _apply_layout(self, imgs):
        # place the imgs by row
        # so the first row is filled,
        # then second, until last

        ncols = self.layout[1]
        nrows = self.layout[0]
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
            store =  self._delta_time_generator
            img, (frame_number, frame_time) = store.get_next_image()
            self._delta_frame_number = frame_number
            self.log_position_along(store)
            imgs.append(img)
            # logger.info(f"Reading frame #{frame_number} at time {frame_time} from {self._delta_time_generator}")
            for store in self._store_list:
                if store is self._delta_time_generator:
                    continue
                else:
                    try:
                        img, (frame_number_, frame_time_) = store._get_image_by_time(frame_time, direction="past")
                        if abs(frame_time_ - frame_time) > self._TOLERANCE:
                            logger.warning(f"Time between frames greater than {self._TOLERANCE}")

                    except Exception as error:
                        logger.warning(f"{store} cannot find a frame at {frame_time} ms")
                        img, (frame_number_, frame_time_)= store.get_image(0)

                    logger.info(f"Reading frame #{frame_number_} at time {frame_time_} from {store}")
                    imgs.append(img)
                    self.log_position_along(store)


        else:
            for store in self._store_list:
                img, (frame_number_, frame_timestamp_)= store._get_image_by_time(
                    store.frame_time + self._delta_time
                )
                imgs.append(img)

        ret = len(imgs) == len(self._store_list)
        return ret, imgs

    def _read_all(self):
        ret, imgs = self._read()
        if ret:
            img = self._apply_layout(imgs)
            return ret, img

        else:
            return False, None


    def _main_store_has_updated(self):
        return self._crossindex.loc[
            self._crossindex["delta_number"] == self._delta_frame_number,
            "update_main"
        ].values.tolist()[0]


    def _read_main_only(self):

        self._delta_frame_number+=1

        if self._main_store_has_updated():
            ret, img = self._main_store.read()

        else:
            ret = True
            img = self._main_store._last_img

        if ret:

            bottom_pad = self.height - img.shape[0]
            right_pad = self.width - img.shape[1]

            img = cv2.copyMakeBorder(
                img,
                top=0,
                bottom=bottom_pad,
                left=0,
                right=right_pad,
                borderType=cv2.BORDER_CONSTANT,
                value=[0, 0, 0]
            )

            return ret, img

        else:
            return False, None



    def toggle(self):
        self._main_only = not self._main_only
        if self._main_only:
            for store in self._stores:
                if store is self._main_store:
                    pass
                else:
                    frame_time = self._main_store.frame_time
                    img, (frame_number, frame_time) = store._get_image_by_time(frame_time)
                    store.get_image(frame_number - 1)


    def read(self, main_only=None):

        if (main_only is None and self._main_only) or main_only is True:
            ret, img = self._read_main_only()
        else:
            ret, img = self._read_all()

        if self._main_store._metadata.get("idtrackerai-color", False):
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        return ret, img



    def release(self):
        for store in self._store_list:
            store.close()

    def close(self):
        self.release()


    def _read_test_frame(self):
        pos_msec = self.get(CAP_PROP_POS_MSEC)
        ret, frame = self.read(main_only=False)
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
            fps = self._delta_time_generator._metadata["framerate"]
            return fps

        elif index == CAP_PROP_POS_MSEC:
            return self._delta_time_generator.frame_time

        elif index == CAP_PROP_FRAME_COUNT:
            last_frame_number = self._delta_time_generator.last_frame_number
            logger.warning("###########################")
            logger.warning(last_frame_number)
            logger.warning("###########################")
            return last_frame_number

        elif index == CAP_PROP_POS_FRAMES:
            return self.delta_frame_number


        else:
            return self._main_store.get(index)

    def set(self, index, value):

        logger.warning(f"Setting {index} to {value}")
        if index in [CAP_PROP_POS_FRAMES, CAP_PROP_POS_MSEC]:
            try:
                self._delta_time_generator.set(index, value, absolute=True)
                self._delta_frame_number = self._delta_time_generator.frame_number
            except Exception as error:
                logger.error(error)
                logger.error(traceback.print_exc())
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
        if self._data_interval is None:

            _, (frame_number, frame_time) = self._delta_time_generator.get_next_image()

            main_store_interval = self._main_store.get_data_interval(what="frame_time", pad=10)
            self._delta_time_generator._set_posmsec(main_store_interval[0], absolute=True)
            begin = self._delta_time_generator.frame_number
            self._delta_time_generator._set_posmsec(main_store_interval[1], absolute=True)
            end = self._delta_time_generator.frame_number
            self._delta_time_generator.get_image(frame_number - 1)

            assert begin < end
            self._data_interval = (begin, end)

        return self._data_interval


    def compute_main_chunk_and_frame_idx(self, crossindex):

        start_index = {v[0]: k for k, v in self._main_store._index.chunk_index["frame_number"].items()}
        chunks = []
        frame_idxs = []
        counter = crossindex["main_number"].values[0]
        chunk =  self._main_store._index.get_chunk_and_frame_idx_from_frame_number(counter)
        last_frame_number = 0


        for frame_number in crossindex["main_number"]:
            if frame_number in start_index:
                chunk = start_index.pop(frame_number)
                if len(chunks) != 0: counter = 0

            chunks.append(chunk)
            frame_idxs.append(counter)
            if last_frame_number != frame_number:
                counter += 1

            last_frame_number = frame_number


        return chunks, frame_idxs

    def get_crossindex(self):
        """
        Returns:
            crossindex (pd.DataFrame): A dataframe as many rows as frames on the longest store and columns:
                delta_number  frame_time  main_number  main_chunk  main_frame_idx  update_main

                * delta_number is the frame_number of the longest store (delta_time_generator)
                * frame_time is the delta_time_generator frame_time
                * main_number is the frame_number in the main store
                * main_chunk is the chunk id in the main store
                * main_frame_idx is the frame_idx in the main store
                * update_main is always False except when a delta time generator time step causes the main_store img to update
        """


        if self._crossindex is None:

            if os.path.exists("cross_index.csv"):
                self._crossindex = pd.read_csv("cross_index.csv", index_col=0)
            else:
                # supported only for two stores
                if len(self._stores) != 2:
                    raise NotImplementedError


                delta_metadata = pd.DataFrame.from_dict(
                    self._delta_time_generator.get_frame_metadata()
                )

                main_metadata = pd.DataFrame.from_dict(
                    self._main_store.get_frame_metadata()
                )

                delta_metadata.columns = ["delta_number", "frame_time"]
                main_metadata.columns = ["main_number", "frame_time"]

                crossindex = pd.merge_asof(
                    delta_metadata, main_metadata,
                    direction="backward",
                    tolerance=1000,
                    left_on="frame_time",
                    right_on="frame_time",
                )


                import ipdb; ipdb.set_trace()

                crossindex=crossindex.loc[
                    ~np.isnan(crossindex["main_number"])
                ]


                chunks, frame_idxs = self.compute_main_chunk_and_frame_idx(crossindex)


                # start_index = {v[0]: k for k, v in self._main_store._index.chunk_index["frame_number"].items()}
                crossindex["main_chunk"] = chunks
                crossindex["main_frame_idx"] = frame_idxs
                crossindex["update_main"] = False
                crossindex.loc[crossindex["main_number"].drop_duplicates().index, "update_main"] = True

                self._crossindex = crossindex

        return self._crossindex


    def export_index_to_csv(self):
        cross_index = self.get_crossindex()
        cross_index.to_csv("cross_index.csv")

