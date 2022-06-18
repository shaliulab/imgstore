import os.path
import pickle
import cv2
try:
    import cv2cuda
except Exception:
    cv2cuda=None
import warnings
from imgstore.constants import SQLITE3_INDEX_FILE
from imgstore.util import ImageCodecProcessor
from imgstore.index import ImgStoreIndex
from imgstore.stores.utils.formats import get_formats

formats=get_formats(cache=True, video=True, directory=True, raw=True)

class PickleMixIn:
    """
    A mixin to make it possible to serialize a store,
    making it possible to pass it through a multiprocessing pipeline
    """
    def __setstate__(self, d):

        self._codec_proc = ImageCodecProcessor()
        if d["_decode_image"]=="function":
            d["_decode_image"] = self._codec_proc.autoconvert
        
        if d["_encode_image"]=="function":
            d["_encode_image"] = self._codec_proc.autoconvert
        
        index_path=os.path.join(
            os.path.dirname(d["_basedir"]),
            SQLITE3_INDEX_FILE
        )
        
        if "_index" in d:
            d["_index"] = ImgStoreIndex.new_from_file(d["_index"])
        
        self.__dict__=d
        
        codec=formats[self._format]

        if cv2cuda is not None and codec == "h264_nvenc":
            video_writer_f = cv2cuda.VideoWriter
            apiPreference="FFMPEG"
        else:
            video_writer_f = cv2.VideoWriter
            apiPreference=cv2.CAP_FFMPEG
        
        if self._mode == "w":
            if "_last_capfn" in self.__dict__:
                self._cap = video_writer_f(
                    filename=self._last_capfn,
                    apiPreference=apiPreference,
                    fourcc=codec,
                    frameSize=self._imgshape[:2][::-1],
                    fps=self._fps,
                    isColor=self._color,
                )
                self._last_capfn=None
        elif self._mode == "r":
            if "_capfn" in self.__dict__:
                self._cap = cv2.VideoCapture(self._capfn)

    def __getstate__(self):

        d = self.__dict__.copy()
        d.pop("_codec_proc")

        if d["_decode_image"] is not None:
            d["_decode_image"]="function"
        if d["_encode_image"] is not None:
            d["_encode_image"]="function"


        if "_cap" in d: d.pop("_cap")
        if "_index" in d:
            d["_index"]=d["_index"].path

        return d


    # def save(self, path):
    #     with open(path, "wb") as filehandle:
    #         d=self.__getstate__()

    #         for e in d:
    #             try:
    #                 pickle.dump(d[e], filehandle)
    #             except Exception as error:
    #                 warnings.warn(f"{e} cannot be pickled")

    def save(self, path):
        with open(path, "wb") as filehandle:
                pickle.dump(self, filehandle)


    @classmethod
    def load(cls, path):
        with open(path, "rb") as filehandle:
            return pickle.load(filehandle)        
