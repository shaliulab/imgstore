import logging
import traceback

import numpy as np

logger = logging.getLogger(__name__)

class CV2Compat:
    """
    Give the imgstore classes an OpenCV-like API
    so they can be exchanged in place of the cv2.VideoCapture()
    without changing any depending code
    """
  
    def read(self):

        try:
            img, (_, _) = self.get_next_image()
            if isinstance(img, np.ndarray):
                ret = True
            else:
                ret = False
                
            return ret, img
        
        except Exception as error:
            logger.error(error)
            logger.error(traceback.print_exc())
            img = None
            return False, img

    def release(self):
        self.close()

    def _set_posmsec(self, new_timestamp, absolute=False):

        if not absolute:
            timestamp_0 = self._index.get_chunk_metadata(self._chunk_n)["frame_time"][0]
            new_timestamp += timestamp_0

        img, (frame_number, timestamp) = self._store._get_image_by_time(new_timestamp)
        self._store.get_image(max(0, frame_number - 1))


    def _get_posmsec(self, absolute=False):
        _, (frame_number, timestamp) = self.get_next_image()
        _ = self.get_image(frame_number-1)

        if absolute:
            posmsec = timestamp
        else:
            timestamp_0 = self._index.get_chunk_metadata(self._chunk_n)["frame_time"][0]
            posmsec = timestamp - timestamp_0
        
        return posmsec

    def _set_posframes(self, posframes, absolute=False):

        if not absolute:
            frame_number_0 = self._index.get_chunk_metadata(self._chunk_n)["frame_number"][0]
            posframes += frame_number_0

        img, (frame_number, timestamp) = self._store.get_image(posframes)
        self._store.get_image(max(0, frame_number - 1))


    def _get_posframes(self, absolute=False):
        _, (frame_number, _) = self.get_next_image()
        _ = self.get_image(frame_number-1)

        if absolute:
           posframes = frame_number
        else:
            frame_number_0 = self._index.get_chunk_metadata(self._chunk_n)["frame_number"][0]
            posframes = frame_number - frame_number_0
       
        return posframes


    def _set_posrel(self, posrel, absolute=False):

        chunk_t0 = self._index.get_chunk_metadata(self._chunk_n)["frame_time"][0]
        chunk_tn = self._index.get_chunk_metadata(self._chunk_n)["frame_time"][-1]
        duration = chunk_tn - chunk_t0
        timestamp = chunk_t0 * posrel * duration
        self._set_posmsec(timestamp, absolute=False)
  
    def _get_posrel(self):
        posframes = self._getposframes()
        framecount = self._get_framecount()
        posframes_rel = posframes / framecount
        return posframes_rel


    def _get_framecount(self):
        return int(self._metadata["chunksize"])

    def _get_framerate(self):
        return int(self._metadata["framerate"])

    def _get_width(self):
        return int(self._metadata["imgshape"][1])

    def _get_height(self):
        return int(self._metadata["imgshape"][0])

    def _get_fourcc(self):
        raise NotImplementedError


    _getters = {
        0: _get_posmsec,
        1: _get_posframes,
        2: _get_posrel,
        3: _get_width,
        4: _get_height,
        5: _get_framerate,
        6: _get_fourcc,
        7: _get_framecount
    }


    _setters = {
        0: _set_posmsec,
        1: _set_posframes,
        2: _set_posrel
    }


    def get(self, index):
        return self._getters[index]()

    def set(self, index, value):
        return self._setters[index](value)