import traceback
import warnings

from idtrackerai.constants import EXTENSIONS
from confapp import conf, load_config
import cv2


config = load_config()

if config.MULTI_STORE_ENABLED:
    import imgstore.stores.multi as imgstore
else:
    import imgstore.stores as imgstore

class VideoCapture():

    def __init__(self, path):

        if type(path) is self.__class__:
            if type(path) is cv2.VideoCapture:
                cap = path
                capture_type = "opencv"
            elif type(path) is imgstore.VideoImgStore:
                cap = path
                capture_type = "imgstore"
        
        elif any(path.endswith(ext) for ext in EXTENSIONS["imgstore"]):
            cap = imgstore.new_for_filename(path)
            cap.get_chunk(config.CHUNK)
            if config.MULTI_STORE_ENABLED:
                if config.SELECTED_STORE:
                    if config.SELECTED_STORE in cap._stores:
                        cap.select_store(config.SELECTED_STORE)
                    else:
                        raise Exception(f"{config.SELECTED_STORE} is not one of the availble stores in {path}")

            capture_type = "imgstore"

        elif any(path.endswith(ext) for ext in EXTENSIONS["video"]):
            cap = cv2.VideoCapture(path)
            capture_type = "opencv"


        self._cap = cap
        self._type = capture_type

    def _get(self, property):
        if property == "NUMBER_OF_FRAMES_IN_CHUNK":
            # this is the number of frames in the chunk
            if self._type == "imgstore":
                return len(self._get_chunk_metadata(self._chunk_n)["frame_number"])
            
            elif self._type == "opencv":
                return self._cap.get(7)

        elif property == "NUMBER_OF_FRAMES":
            # this is the number of frames in the store up to the last one
            # in the current chunk
            if self._type == "imgstore":
                return self._get("ENDING_FRAME_OF_CHUNK")+1
            elif self._type == "opencv":
                return self._get(7)


        elif property == "STARTING_FRAME_OF_CHUNK":
            # this is the frame number of the last frame in this chumk
            if self._type == "imgstore":
                return self._get_chunk_metadata(self._chunk_n)["frame_number"][0]
            
            elif self._type == "opencv":
                return self._cap.get(1)

        elif property == "ENDING_FRAME_OF_CHUNK":
            # this is the frame number of the last frame in this chumk
            if self._type == "imgstore":
                return self._get_chunk_metadata(self._chunk_n)["frame_number"][-1]
            
            elif self._type == "opencv":
                return self._cap.get(7)

        elif property == "TOTAL_NUMBER_OF_FRAMES":
            # this is the frame number of the last frame in this store
            if self._type == "imgstore":
                return self._cap._index._summary("frame_max")

            elif self._type == "opencv":
                return self._cap.get(7)

        
        else:
            raise Exception(f"Property not found")

    def _set(self, property, value):
        if property == "CAP_PROP_POS_FRAMES":
            return self._cap.set(property, value)
        
        else:
            raise Exception(f"Property not found")

    def get(self, property):

        if type(property) is str:
            return self._get(property)
        else:
            return self._cap.get(property)


    def set(self, property, value):

        if type(property) is str:
            return self._set(property, value)
        else:
            return self._cap.set(property, value)


    def __getattr__(self, __name: str):
        return getattr(self._cap, __name)