import warnings
import logging
import numpy as np
import cv2

from imgstore.stores.utils.mixins.extract import _extract_store_metadata
from imgstore.stores.core import new_for_filename as new_for_filename_single
from imgstore.stores.utils.mixins.cm import ContextManagerMixin
logger = logging.getLogger(__name__)
_VERBOSE_DEBUG_GETS = False

### Multistore module
#
# The master store frame always goes on the left
# The selected store is the one that sets the time series i.e. the frame times available

class MultiStore(ContextManagerMixin):

    def __init__(self, **stores):
        self._stores = stores
        self._master = stores["master"]
        self._selected = None
        self._index = None

    def close(self):
        for store_name, store in self._stores.items():
            store.close()


    def _apply_layout(imgs):
        
        shape = imgs[0].shape
        reshaped_imgs=[]

        for img in imgs:

            if len(shape) > len(img.shape):
                img = np.stack([img, ] * 3, axis=2)
            elif len(shape) < len(img.shape):
                img = img[:, :, 0]

            reshaped_imgs.append(
                cv2.resize(img, shape[:2][::-1], cv2.INTER_NEAREST)
            )
        
        img=np.stack(reshaped_imgs, axis=1)
        return img

    
    def select_store(self, name):

        try:
            self._selected_name = name
            self._selected = self._stores[name]
            self._metadata = self._selected._metadata.copy()
            self._metadata["imgshape"] = (self._master._metadata["imgshape"][0], self._master._metadata["imgshape"][1]*len(self._stores))
            self._index = self._selected._index
        except KeyError:
            warnings.warn(f"{name} is not a store", stacklevel=2)
    
    def get_next_image(self):

        _, meta = self._selected.get_next_image()
        
        frame_number, frame_time = meta
        
        master_img, _ = self._master.get_nearest_image(frame_time)

        imgs = [master_img]

        for store_name, store in self._stores.items():
            if store_name == "master":
                continue
            else:
                img, _ = store.get_nearest_image(frame_time)
                imgs.append(img)
        
        img = self._apply_layout(imgs)
        return img, meta


    def get_next_framenumber(self):

        frame_number, idx = self._selected.get_next_framenumber()
        return frame_number, idx

    def get_nearest_image(self, frame_time):

        imgs = []
        master_img, selected_meta = self._master.get_nearest_image(frame_time)

        for store_name, store in self._stores.items():
            if store_name == "master":
                continue
            else:
                img, meta = store.get_nearest_image(frame_time)
                if store_name == self._selected_name:
                    selected_meta = meta

            imgs.append(img)

        img = self._apply_layout(imgs)
        return img, selected_meta

    def get_image(self, frame_number, exact_only=True, frame_index=None):
        """
        seek to the supplied frame_number or frame_index. If frame_index is supplied get that image,
        otherwise get the image corresponding to frame_number

        :param frame_number:  (frame_min, frame_max)
        :param exact_only: If False return the nearest frame
        :param frame_index: frame_index (0, frame_count]
        """

        if _VERBOSE_DEBUG_GETS:
            self._log.debug('get_image %s (exact: %s) frame_idx %s' % (frame_number, exact_only, frame_index))

        data = self._index.find_all("frame_number", frame_number, exact_only=exact_only)
        # TODO How do we get the frame_time from data?
        frame_time = data[1]

        return self.get_nearest_image(frame_time)


def new_for_filename(path, **kwargs):  
    stores = {"master": new_for_filename_single(path, **kwargs)}
 
    metadata = _extract_store_metadata(path)
    for cameras in metadata.get("extra_cameras", []):
        stores[cameras] = new_for_filename_single(cameras, **kwargs)

    multistore = MultiStore(**stores)
    return multistore


def new_for_format():
    raise NotImplementedError()

__all__= ["new_for_filename", "new_for_format"]
