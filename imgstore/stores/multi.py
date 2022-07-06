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
        self._crossindex_pointer =0

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
            self._stores["selected"] = self._selected
        except KeyError:
            warnings.warn(f"{name} is not a store", stacklevel=2)

        
    def get_next_image_single_store(self, store_name, fn):
        if (self._stores[store_name].frame_number + 1) == fn:
            with codetiming.Timer(text="get next image on master took {:.8f} seconds to compute", logger=logger.debug):
                img, meta = self._stores[store_name].get_next_image()
        elif (self._stores[store_name].frame_number) == fn:
            with codetiming.Timer(text="_last_img.copy() on master took {:.8f} seconds to compute", logger=logger.debug):
                img = self._stores[store_name]._last_img.copy()
                meta = (self._stores[store_name].frame_number, self._stores[store_name].frame_time)
        else:
            with codetiming.Timer(text="get image on master took {:.8f} seconds to compute", logger=logger.debug):
                try:
                    img, meta = self._stores[store_name].get_image(fn)
                except Exception as error:
                    raise error

        logger.debug(f"{store_name} is in frame number #{self._stores[store_name].frame_number}")
        return img, meta

    
    def get_next_image(self):
        
        id=self._crossindex_pointer+1

        master_fn = self.crossindex.find_fn_given_id(store="master", id=id)
        selected_fn = self.crossindex.find_fn_given_id(store="selected", id=id)
        
        master_img, master_meta = self.get_next_image_single_store("master", master_fn)
        selected_img, selected_meta = self.get_next_image_single_store("selected", selected_fn)

        imgs=[master_img, selected_img]

        with codetiming.Timer(text="_apply_layout took {:.8f} seconds to compute", logger=logger.debug):
            img = self._apply_layout(imgs)
        
        meta = (id, selected_meta[1])
        self._crossindex_pointer =id
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
        # TODO
        # Can the idx be None

        frame_number, idx = (self._crossindex_pointer, None)
        return frame_number, idx

    def get_nearest_image(self, frame_time):
        try:
            master_img, master_meta = self._master.get_nearest_image(frame_time, future=False)
        except TypeError:
            logger.warning(f"Cannot fetch a frame in master before {frame_time}. Fetching from future")
            master_img, master_meta = self._master.get_nearest_image(frame_time)

        selected_meta = master_meta

        self._crossindex_pointer = self.crossindex.find_id_given_fn("master", master_meta[0])
        selected_frame_number=self.crossindex.find_selected_fn(master_meta[0])

        imgs = [master_img]

        for store_name, store in self._stores.items():
            if store_name == "master":
                continue
            else:
                img, meta = store.get_image(selected_frame_number)
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

        id=frame_number
        del frame_number

        if _VERBOSE_DEBUG_GETS:
            self._log.debug('get_image %s (exact: %s) frame_idx %s' % (id, exact_only, frame_index))
        
        imgs=[None, None]
        with codetiming.Timer(text="find_master_fn took {:.8f} seconds to compute", logger=logger.debug):
            master_fn = self.crossindex.find_fn_given_id("master", id)


        with codetiming.Timer(text="get_image on master took {:.8f} seconds to compute", logger=logger.debug):
            imgs[0], _ = self._master.get_image(master_fn)
        with codetiming.Timer(text="get_image on selected took {:.8f} seconds to compute", logger=logger.debug):
            selected_fn = self.crossindex.find_fn_given_id("selected", "id") 
            imgs[1], selected_meta = self._selected.get_image(selected_fn)


        img = self._apply_layout(imgs)
        self._crossindex_pointer = id
        return img, selected_meta


    def get_chunk(self, chunk):
        self._log.debug(f"{self}.get_chunk({chunk})")
        _, (master_fn, master_ft) = self._master.get_chunk(chunk)

        for store_name, store in self._stores.items():
            if store_name == "master":
                continue
            else:
                with codetiming.Timer():
                    id=self.crossindex.find_id_given_fn("master", master_fn)
                    selected_fn=self.crossindex.find_fn_given_id("selected", id)
                    img, (fn, ft) = self.get_image(selected_fn)
                # This is a bit slower
                # with codetiming.Timer():
                #     img2, (fn2, ft2) = self.get_nearest_image(master_ft)


                store.get_image(max(0, fn))
        self._crossindex_pointer = self.crossindex.find_id_given_fn("master", master_fn)


    @classmethod
    def new_for_filename(cls, *args, **kwargs):
        return new_for_filename(*args, **kwargs)

    def set(self, key, value):
        result=self._selected.set(key, value)
        if key == 0:
            self.get_nearest_image(value-1)
        elif key == 1:
            self._crossindex_pointer=value
            self.get_image(max(value-1, 0))

        return result


    def get(self, property):

        if property == "NUMBER_OF_FRAMES_IN_CHUNK":
            # this is the number of frames in the chunk
            chunk = self._master._chunk_n
            return self.crossindex.get_number_of_frames_in_chunk(chunk)
            

        elif property == "NUMBER_OF_FRAMES":
            # this is the number of frames in the store up to the last one
            # in the current chunk
            raise NotImplementedError()


        elif property == "STARTING_FRAME_OF_CHUNK":
            # this is the frame number of the last frame in this chumk
            chunk = self._master._chunk_n
            return self.crossindex.get_starting_frame_of_chunk(chunk)
         

        elif property == "ENDING_FRAME_OF_CHUNK":
            # this is the frame number of the last frame in this chumk
            chunk = self._master._chunk_n
            return self.crossindex.get_ending_frame_of_chunk(chunk)

        elif property == "TOTAL_NUMBER_OF_FRAMES":
            return self.crossindex.get_number_of_frames()

        elif property == 1:
            return self._crossindex_pointer

        else:
            return self._selected.get(property)

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
