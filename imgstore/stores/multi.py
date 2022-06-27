"""
Multistore module

The master store frame always goes on the left
The selected store is the one that sets the time series i.e. the frame times available
"""

import warnings
import os.path
import traceback
import logging
import numpy as np
import cv2
import codetiming
from imgstore.stores.utils.mixins.extract import _extract_store_metadata
from imgstore.stores.core import new_for_filename as new_for_filename_single
from imgstore.stores.utils.mixins import ContextManagerMixin
from imgstore.stores.utils.mixins.multi import MultiStoreCrossIndexMixIn
from confapp import conf, load_config
from imgstore import constants
config = load_config(constants)

logger = logging.getLogger(__name__)
_VERBOSE_DEBUG_GETS = False


class VideoImgStore(ContextManagerMixin, MultiStoreCrossIndexMixIn):

    def __init__(self, **stores):
        self._stores = stores
        self._master = stores["master"]
        self._selected = stores.get("lowres/metadata.yaml", stores["master"])
        self._selected_name = "master"
        self._index = self._master._index
        self._basedir = self._master._basedir

    def close(self):
        for store_name in self._stores:
            self._stores[store_name].close()

    @property
    def _capfn(self):
        return self._master._capfn

    @property
    def full_path(self):
        return self._master.full_path

    @property
    def _chunk_n(self):
        return self._master._chunk_n

    @staticmethod
    def _apply_layout(imgs):

        assert len(imgs) >= 1

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
       
        img=np.concatenate(reshaped_imgs, axis=1)

        if getattr(config, "COLOR", False) and len(img.shape) == 2:
            logger.debug(f"Converting grayscale image of shape {img.shape} to BGR")
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


        return img

    
    def select_store(self, name):

        try:
            self._selected_name = name
            self._selected = self._stores[name]
            self._metadata = self._selected._metadata.copy()
            self._metadata["imgshape"] = (self._master._metadata["imgshape"][0], self._master._metadata["imgshape"][1]*len(self._stores))
            self._index = self._selected._index
            self._make_crossindex()
        except KeyError:
            warnings.warn(f"{name} is not a store", stacklevel=2)
    
    def get_next_image(self):

        img, meta = self._selected.get_next_image()
        master_fn = self.crossindex.find_master_fn(self._selected.frame_number)
        frame_number, frame_time = meta

        if conf.LOOKUP_NEAREST:
            with codetiming.Timer(text="get nearest image on master took {:.8f} seconds to compute", logger=print):
                master_img, _ = self._master.get_nearest_image(frame_time)
        else:
            if (self._master.frame_number + 1) == master_fn:
                with codetiming.Timer(text="get next image on master took {:.8f} seconds to compute", logger=print):
                    master_img, _ = self._master.get_next_image()
            elif (self._master.frame_number) == master_fn:
                with codetiming.Timer(text="_last_img.copy() on master took {:.8f} seconds to compute", logger=print):
                    master_img = self._master._last_img.copy()
            else:
                with codetiming.Timer(text="get image on master took {:.8f} seconds to compute", logger=print):
                    try:
                        master_img, _ = self._master.get_image(master_fn)
                    except Exception as error:
                        raise error

        print(f"Master is in frame number #{self._master.frame_number}")
        print(f"Selected is  in frame number #{self._selected.frame_number}")



        if self._selected_name == "master":
            imgs = [master_img]
        else:
            imgs = [master_img, img]

        for store_name, store in self._stores.items():
            if store_name in ["master", self._selected_name]:
                continue
            else:
                with codetiming.Timer(text="get nearest image on {store_name} took {:.8f} seconds to compute", logger=print):
                    img, _ = store.get_nearest_image(frame_time)
                    imgs.append(img)

        with codetiming.Timer(text="_apply_layout took {:.8f} seconds to compute", logger=print):
            img = self._apply_layout(imgs)
        return img, meta

    
    def read(self):

        img, _ = self.get_next_image()

        if type(img) is np.ndarray:
            ret = True
        else:
            img = None
            ret = False
        
        return ret, img


    def get_next_framenumber(self):

        frame_number, idx = self._selected.get_next_framenumber()
        return frame_number, idx

    def get_nearest_image(self, frame_time):

        master_img, selected_meta = self._master.get_nearest_image(frame_time)
        imgs = [master_img]

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
        
        imgs=[None, None]
        # master_fn = self.crossindex.loc[frame_number, ("master", "frame_number")]
        master_fn = self.crossindex.find_master_fn(frame_number)

        imgs[0], _ = self._master.get_image(master_fn)
        imgs[1], selected_meta = self._selected.get_image(master_fn)

        img = self._apply_layout(imgs)
        return img, selected_meta

        # chunk, frame_idx, frame_number, frame_time = self._index.find_all("frame_number", frame_number, exact_only=exact_only)
        # return self.get_nearest_image(frame_time)


    def get_chunk(self, chunk):
        self._log.debug(f"{self}.get_chunk({chunk})")
        _, (fn, ft) = self._master.get_chunk(chunk)

        for store_name, store in self._stores.items():
            if store_name == "master":
                continue
            else:
                img, (fn, ft) = store.get_nearest_image(ft)
                store.get_image(max(0, fn))


    @classmethod
    def new_for_filename(cls, *args, **kwargs):
        return new_for_filename(*args, **kwargs)

    def __getattr__(self, __name: str):
        return getattr(self._selected, __name)


def load_extra_cameras(path, **kwargs):
    """
    Load the stores listed in metadata.yaml (available in path) under key extra_cameras,
    and return them as a dictionary where the keys are the listed paths

    Arguments:

        * path (str): Path to an imgstore folder or the metadata.yaml inside it
        **kwargs (dict): Arguments to new_for_filename_single
    """

    METADATA_KEY="extra_cameras"

    stores = {}

    metadata = _extract_store_metadata(path)
    for camera in metadata.get(METADATA_KEY, []):
        try:
            stores[camera] = new_for_filename_single(camera, **kwargs)
            print(f"Loaded {camera} imgstore")

        except FileNotFoundError:
            if os.path.isfile(path):
                master_path = os.path.dirname(path)
            else:
                master_path = path

            find_path_to_camera = os.path.join(master_path, camera)
            assert os.path.exists(find_path_to_camera), f"{find_path_to_camera} does not exist"
            stores[camera] = new_for_filename_single(find_path_to_camera, **kwargs)

    return stores

def new_for_filename(path, **kwargs):
    stores = {"master": new_for_filename_single(path, **kwargs)}
    print("Loaded master imgstore")
    stores.update(load_extra_cameras(path, **kwargs))
    multistore = VideoImgStore(**stores)
    return multistore


def new_for_format():
    raise NotImplementedError()

__all__= ["new_for_filename", "new_for_format"]
