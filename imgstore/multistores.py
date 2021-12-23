import numpy as np
import cv2

from .stores import new_for_filename


class MultiStore:

    _LAYOUT = (2, 1)

    def __init__(self, store_list, layout=None, adjust_by="pad", **kwargs):

        self._stores = {
            path: new_for_filename(path, **kwargs) for path in store_list
        }

        if layout is None:
            self._layout = self._LAYOUT

        self._adjust_by = adjust_by

        self._refstore = self._stores[store_list[0]]

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

        width = rows[0].shape[1]

        for row in rows[1:]:
            width_diff = int(row.shape[1] - width)
            if width_diff != 0:
                if self._adjust_by == "pad":

                    if width_diff % 2 == 0:
                        odd_pixel = 0
                    else:
                        odd_pixel = 1

                    row = cv2.copyMakeBorder(
                        row,
                        0,
                        0,
                        width_diff / 2,
                        width_diff / 2 + odd_pixel,
                        cv2.BORDER_CONSTANT,
                        value=0,
                    )

                elif self._adjust_by == "resize":
                    row = cv2.resize(
                        row, (width, row.shape[0]), cv2.INTER_AREA
                    )

            assert row.shape[1] == width

        return np.vstack(rows)

    @staticmethod
    def make_row(imgs):

        height = imgs[0].shape[0]
        for img in imgs[1:]:
            assert img.shape[0] == height

        return np.hstack(imgs)

    def _read(self):

        imgs = []
        for store in self._stores.values():
            ret, img = store.read()
            imgs.append(img)

        return imgs

    def read(self):
        self._apply_layout(self._read())

    def release(self):
        for store in self._stores.values():
            store.close()

    def close(self):
        self.release()

    def get(self, index):
        return self._refstore.get(index)

    def set(self, index, value):

        for store in self._stores.values():
            store.set(index, value)

    def get_image(self, frame_number):
        raise Exception("get_image method is not meaningful in a multistore")

    def get_image_by_time(self, timestamp):

        imgs = []
        for store in self._stores.values():
            img, _ = store._get_image_by_time(timestamp)
            imgs.append(img)

        return imgs

    def __getattr__(self, attr):
        return getattr(self._refstore, attr)


def new_for_filenames(store_list, **kwargs):
    return MultiStore(store_list, **kwargs)
