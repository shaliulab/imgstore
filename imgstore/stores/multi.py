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
from imgstore.util import ensure_color, ensure_grayscale
from imgstore import constants

logger = logging.getLogger(__name__)
_VERBOSE_DEBUG_GETS = False


class VideoImgStore(ContextManagerMixin, MultiStoreCrossIndexMixIn):

    def __init__(self, main_, secondary_, **stores):
        

        self._config = load_config(constants)
        self._crossindex=None
        self.main=main_
        self.secondary=secondary_

        self._stores = stores
        self._crossindex_pointer =0
        self.is_multistore = True

    @property
    def _index(self):
        return self._stores["selected"]._index
    
    @property
    def _basedir(self):
        return self._stores[self.main]._basedir

    def close(self):
        for store_name in self._stores:
            self._stores[store_name].close()
            
    def get_root(self):
        return self._stores[self.main]._basedir

    @property
    def _capfn(self):
        return self._stores[self.main]._capfn

    @property
    def full_path(self):
        return self._stores[self.main].full_path

    @property
    def _chunk_n(self):
        return self._stores[self.main]._chunk_n

    def _apply_layout(self, imgs):

        assert len(imgs) >= 1
        if self._config.FIRST_FEED == "master":
            pass
        elif self._config.FIRST_FEED == "selected":
            imgs=imgs[::-1]
        else:
            raise Exception("Please specify which feed should be shown to the left. Either master or selected")

        shape = imgs[0].shape
        shapes = np.vstack([img.shape for img in imgs])
        max_height, max_width = shapes.max(0)

        reshaped_imgs=[]

        for img in imgs:

            if len(shape) == 3:
                img = ensure_color(img)
            if len(shape) == 2:
                img = ensure_grayscale(img)


            if self._config.RESHAPE_METHOD == "resize":
                reshaped_img = cv2.resize(
                    img,
                    shape[::-1],
                    cv2.INTER_AREA
                )

            elif self._config.RESHAPE_METHOD == "pad":
                reshaped_img = cv2.copyMakeBorder(
                    img,
                    top=0,
                    bottom=max(0, max_height-img.shape[0]),
                    right=max(0, max_width-img.shape[1]),
                    left=0,
                    borderType=cv2.BORDER_CONSTANT,
                    value=255
                )

            reshaped_imgs.append(
                reshaped_img
            )
        img=np.concatenate(reshaped_imgs, axis=1)

        if getattr(self._config, "COLOR", False) and len(img.shape) == 2:
            logger.debug(f"Converting grayscale image of shape {img.shape} to BGR")
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        return img

    
    def select_store(self, name):

        # NOTE:
        # Please refactor code everywhere so master and selected
        # are not linked to high res and high speed
        if name == "highres":
            return

        try:
            print(f"Selecting store {name}")
            self._stores["selected"] = self._stores[name]
            self._metadata = self._stores["selected"]._metadata.copy()
            self._metadata["imgshape"] = (
                self._stores[self._config.FIRST_FEED]._metadata["imgshape"][0],
                self._stores[self._config.FIRST_FEED]._metadata["imgshape"][1] * len(self._stores)
            )
            self._make_crossindex()
            self._stores["selected"] = self._stores["selected"]
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

        imgs = [master_img, selected_img]

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
            master_img, master_meta = self._stores["master"].get_nearest_image(frame_time, future=False)
        except TypeError:
            logger.warning(f"Cannot fetch a frame in master before {frame_time}. Fetching from future")
            master_img, master_meta = self._stores["master"].get_nearest_image(frame_time)

        selected_meta = master_meta

        self._crossindex_pointer = self.crossindex.find_id_given_fn("master", master_meta[0])
        selected_frame_number=self.crossindex.find_selected_fn(master_meta[0])

        imgs = [master_img]

 
        selected_img, selected_meta = self._stores["selected"].get_image(selected_frame_number)
        imgs.append(selected_img)
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

        selected_meta, imgs = self.get_images(frame_number, exact_only, frame_index)
        img = self._apply_layout(imgs)
        self._crossindex_pointer = frame_number
        return img, selected_meta


    def get_chunk(self, chunk):
        self._log.debug(f"{self}.get_chunk({chunk})")
        _, (fn, ft) = self._stores[self.main].get_chunk(chunk)

        with codetiming.Timer():
            id=self.crossindex.find_id_given_fn(self.main, fn)
            selected_fn=self.crossindex.find_fn_given_id(self.secondary, id)
            img, (fn, ft) = self.get_image(id)
            self._stores[self.secondary].get_image(max(0, selected_fn))
        self._crossindex_pointer = id


    def get_images(self, frame_number, exact_only=True, frame_index=None):
        id=frame_number
        del frame_number

        if _VERBOSE_DEBUG_GETS:
            self._log.debug('get_image %s (exact: %s) frame_idx %s' % (id, exact_only, frame_index))
        
        imgs=[None, None]
        with codetiming.Timer(text="find_master_fn took {:.8f} seconds to compute", logger=logger.debug):
            master_fn = self.crossindex.find_fn_given_id("master", id)

        with codetiming.Timer(text="get_image on master took {:.8f} seconds to compute", logger=logger.debug):
            imgs[0], _ = self._stores["master"].get_image(master_fn)
        with codetiming.Timer(text="get_image on selected took {:.8f} seconds to compute", logger=logger.debug):
            selected_fn = self.crossindex.find_fn_given_id("selected", id) 
            imgs[1], selected_meta = self._stores["selected"].get_image(selected_fn)

        return selected_meta, imgs


    @classmethod
    def new_for_filename(cls, *args, **kwargs):
        return new_for_filename(*args, **kwargs)

    def set(self, key, value):
        result=self._stores["selected"].set(key, value)
        if key == 0:
            self.get_nearest_image(value-1)
        elif key == 1:
            self._crossindex_pointer=value
            self.get_image(max(value-1, 0))

        return result


    def get(self, property):

        if property == "NUMBER_OF_FRAMES_IN_CHUNK":
            # this is the number of frames in the chunk
            chunk = self._stores[self.main]._chunk_n
            return self.crossindex.get_number_of_frames_in_chunk(chunk)
            

        elif property == "NUMBER_OF_FRAMES":
            # this is the number of frames in the store up to the last one
            # in the current chunk
            raise NotImplementedError()


        elif property == "STARTING_FRAME_OF_CHUNK":
            # this is the frame number of the last frame in this chumk
            chunk = self._stores[self.main]._chunk_n
            return self.crossindex.get_starting_frame_of_chunk(chunk)
         

        elif property == "ENDING_FRAME_OF_CHUNK":
            # this is the frame number of the last frame in this chumk
            chunk = self._stores[self.main]._chunk_n
            return self.crossindex.get_ending_frame_of_chunk(chunk)

        elif property == "TOTAL_NUMBER_OF_FRAMES":
            return self.crossindex.get_number_of_frames()

        elif property == 1:
            return self._crossindex_pointer

        else:
            return self._stores["selected"].get(property)

    def __getattr__(self, k: str):
        return getattr(self._stores[self.main], k)



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
    root_dir = os.path.dirname(path)
    cameras = metadata.get(METADATA_KEY, {})
    if isinstance(cameras, list):
        raise Exception(f"extra_cameras attribute in {path} must be a dictionary")
    for camera_name in cameras:
        camera = cameras[camera_name]
        stores[camera_name] = new_for_filename_single(os.path.join(root_dir, camera), **kwargs)
        print(f"Loaded {camera} imgstore")

    return stores

def new_for_filename(path, **kwargs):
    stores = {"master": new_for_filename_single(path, **kwargs)}
    print("Loaded master imgstore")
    stores.update(load_extra_cameras(path, **kwargs))
    metadata = _extract_store_metadata(path)
    stores[metadata["name"]] = stores["master"]
    stores["main"] = stores["master"]
    main = "master"
    secondary = "selected"

    # Set the selected store to be anything other than master
    if "selected" not in stores:
        store_names = list(stores.keys())
        store_names.pop(store_names.index("master"))
        if len(store_names) > 0:
            selected_store = store_names[0]
            stores["selected"]=stores[selected_store]
        else:
            warnings.warn(f"No extra_cameras found in {os.path.realpath(stores['master']._basedir)}")

    # NOTE
    # I have assumed in many parts of the c ode that selected is the high_speed feed
    # and master is the high_res feed.
    # This means if I try to read an high_speed imgstore,
    # I need to swap the identity of master and selected
    if "lowres" in os.path.realpath(stores["master"]._basedir) and \
        not "lowres" in os.path.realpath(stores["selected"]._basedir):
        stores["_master"] = stores["selected"]
        stores["_selected"] = stores["master"]
        stores["selected"] = stores["_selected"]
        stores["master"] = stores["_master"]

        del stores["_master"]
        del stores["_selected"]
        main = "selected"
        secondary = "master"

    multistore = VideoImgStore(main, secondary, **stores)
    return multistore


def new_for_format():
    raise NotImplementedError()

__all__= ["new_for_filename", "new_for_format"]
