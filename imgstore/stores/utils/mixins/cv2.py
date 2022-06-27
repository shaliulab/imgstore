import logging
import traceback
import cv2
import numpy as np
from confapp import conf, load_config
logger = logging.getLogger(__name__)
from imgstore import constants
config = load_config(constants)

class CV2Mixin:
    """
    Give the imgstore classes an OpenCV-like API
    so they can be exchanged in place of the cv2.VideoCapture()
    without changing any depending code

    Frame count and frame time are relative to the first frame of the chunk
    If you want to refer them to the first frame of the whole imgstore, please pass absolute=True
    """

    def read(self):

        try:
            img, (_, _) = self.get_next_image()

            if isinstance(img, np.ndarray):
                ret = True
                if getattr(config, "COLOR", False) and len(img.shape) == 2:
                    logger.debug(f"Converting grayscale image of shape {img.shape} to BGR")
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

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

    def _set_posmsec(self, new_timestamp, absolute=True, **kwargs):

        if not absolute:
            timestamp_0 = self._index.get_chunk_metadata(self._chunk_n)[
                "frame_time"
            ][0]
            new_timestamp += timestamp_0

        img, (frame_number, timestamp) = self.get_nearest_image(
            new_timestamp-1, future=False, **kwargs
        )

    def _get_posmsec(self, absolute=True):

        if absolute:
            return self._index.find_all(what="frame_number", value=self.frame_number+1)[3]

        else:
            timestamp_0 = self._index.get_chunk_metadata(self._chunk_n)[
                "frame_time"
            ][0]
            posmsec = self.frame_time - timestamp_0

        return posmsec

    def _set_posframes(self, posframes):

        # frame_number_0 = self._get_chunk_metadata(self._chunk_n)[
        #     "frame_number"
        # ][0]
        # posframes += frame_number_0

        if posframes - 1 < 0:
            posframes_f = 0
        else:
            posframes_f = posframes - 1

        img, (frame_number, timestamp) = self.get_image(posframes_f)
        if posframes_f == 0:
            self._cap.set(getattr(cv2, "CAP_PROP_POS_FRAMES", 1), 0)

    def _get_posframes(self, absolute=True):

        if absolute:
            return self.frame_number+1
        else:
            frame_number_0 = self._get_chunk_metadata(self._chunk_n)["frame_number"][0]
            posframes = self.frame_number - frame_number_0+1

        return posframes

    def _set_posrel(self, posrel, absolute=True):

        chunk_t0 = self._index.get_chunk_metadata(self._chunk_n)["frame_time"][
            0
        ]
        chunk_tn = self._index.get_chunk_metadata(self._chunk_n)["frame_time"][
            -1
        ]
        duration = chunk_tn - chunk_t0
        timestamp = chunk_t0 * posrel * duration
        self._set_posmsec(timestamp, absolute=True)

    def _get_posrel(self):
        posframes = self._get_posframes()
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
        7: _get_framecount,
    }

    _setters = {0: _set_posmsec, 1: _set_posframes, 2: _set_posrel}

    def get(self, index):
        return self._getters[index](self)

    def set(self, index, value, **kwargs):
        return self._setters[index](self, value, **kwargs)