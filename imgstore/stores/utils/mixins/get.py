import warnings
import logging
import os.path
import numpy as np
import codetiming
from imgstore.constants import VERBOSE_DEBUG_CHUNKS


_VERBOSE_DEBUG_GETS = False
logger = logging.getLogger(__name__)


class GetTrajectoryMixin:
    
    
    def get_centroid(self, chunk, frame_idx):
        centroid = np.load(os.path.join(self._basedir, str(chunk).zfill(6) + ".npy"))[frame_idx]
        return centroid
    
    def get_data(self, frame_number, **kwargs):
        img, meta = self.get_image(frame_number=frame_number, **kwargs)
        chunk, frame_idx = self._index.find_chunk("frame_number", frame_number)
        centroid = self.get_centroid(chunk, frame_idx)
        meta = list(meta)
        meta.append(centroid)
        meta=tuple(meta)
        return img, meta
        
    def get_next_data(self):
        img, meta = self.get_next_image()
        frame_number = meta[0]
        chunk, frame_idx = self._index.find_chunk("frame_number", frame_number)
        centroid = self.get_centroid(chunk, frame_idx)
        meta = list(meta)
        meta.append(centroid)
        meta=tuple(meta)
        return img, meta


class GetMixin(GetTrajectoryMixin):

    def get_next_image(self):
        frame_number, idx = self._get_next_framenumber_and_chunk_frame_idx()
        assert idx is not None

        if _VERBOSE_DEBUG_GETS:
            self._log.debug('get_next_image frame_number: %s idx %s' % (frame_number, idx))

        return self._get_image(chunk_n=None,  # chunk was already loaded in _get_next_framenumber_and_chunk_frame_idx
                               frame_idx=idx)

    def get_next_framenumber(self):
        return self._get_next_framenumber_and_chunk_frame_idx()[0]

    def get_nearest_image(self, frame_time, **kwargs):
        """
        :param frame_time: (int) Time of a given frame, in ms
        """
        chunk_n, frame_idx = self._index.find_chunk_nearest('frame_time', frame_time, **kwargs)
        return self._get_image(chunk_n, frame_idx)

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
        if frame_index is not None:
            return self._get_image_by_frame_index(frame_index)
        else:
            if (self.frame_number + 1) == frame_number:
                return self.get_next_image()
            elif self.frame_number == frame_number:
                return self._last_img.copy(), (self.frame_number, self.frame_time)
            else:
                return self._get_image_by_frame_number(frame_number, exact_only=exact_only)


    def _get_next_framenumber_and_chunk_frame_idx(self):
        idx = self._chunk_current_frame_idx + 1

        try:
            # try fast to see if next frame is in currently loaded chunk
            frame_number = self._chunk_md['frame_number'][idx]
        except IndexError:
            # open the next chunk
            next_chunk = self._chunk_n + 1
            if next_chunk not in self._index.chunks:
                logger.error("Cannot read chunk %s", next_chunk)
                raise EOFError

            self._load_chunk(next_chunk)

            # first frame is start of chunk
            idx = 0
            frame_number = self._chunk_md['frame_number'][idx]

        return frame_number, idx


    def _get_image(self, chunk_n, frame_idx):
        if chunk_n is not None:
            with codetiming.Timer(text="Loading chunk took {milliseconds:.0f} ms", logger=logger.debug):
                self._load_chunk(chunk_n)
            chunk_n = self._chunk_n

        # ensure the read works before setting frame_number
        with codetiming.Timer(text="Loading image took {milliseconds:.0f} ms", logger=logger.debug):
            _img, (_frame_number, _frame_timestamp) = self._load_image(frame_idx)
            if _img is None:
                warnings.warn(f"Cannot read chunk {chunk_n} frame_idx {frame_idx}. Skipping to chunk {self._chunk_n+1}")
                return self._get_image(self._chunk_n+1, 0)
        with codetiming.Timer(text="Decoding image took {milliseconds:.0f} ms", logger=logger.debug):
            img = self._decode_image(_img)
        self._chunk_current_frame_idx = frame_idx
        self.frame_number = _frame_number
        self.frame_time = _frame_timestamp

        return img, (_frame_number, _frame_timestamp)


    def _get_image_by_frame_index(self, frame_index):
        """
        return the frame at the following index in the store
        """
        if frame_index < 0:
            raise ValueError('seeking to negative index not supported')

        if VERBOSE_DEBUG_CHUNKS:
            self._log.debug('seek by frame_index %s' % frame_index)

        chunk_n, frame_idx = self._index.find_chunk('index', frame_index)

        if chunk_n == -1:
            raise ValueError('frame_index %s not found in index' % frame_index)

        self._log.debug('seek found in chunk %d attempt read chunk index %d' % (self._chunk_n, frame_idx))

        return self._get_image(chunk_n, frame_idx)

    def _get_image_by_frame_number(self, frame_number, exact_only):
        if VERBOSE_DEBUG_CHUNKS:
            self._log.debug('seek by frame_number %s (exact: %s)' % (frame_number, exact_only))

        if exact_only:
            chunk_n, frame_idx = self._index.find_chunk('frame_number', frame_number)
        else:
            chunk_n, frame_idx = self._index.find_chunk_nearest('frame_number', frame_number)

        if chunk_n == -1:
            raise ValueError('frame #%s not found in any chunk' % frame_number)

        return self._get_image(chunk_n, frame_idx)

    
    def _get_image_by_time(self, frame_time, exact_only):
        if VERBOSE_DEBUG_CHUNKS:
            self._log.debug('seek by frame_time %s (exact: %s)' % (frame_time, exact_only))

        if exact_only:
            chunk_n, frame_idx = self._index.find_chunk('frame_time', frame_time)
        else:
            chunk_n, frame_idx = self._index.find_chunk_nearest('frame_time', frame_time)

        if chunk_n == -1:
            raise ValueError('frame_time #%s not found in any chunk' % frame_time)

        return self._get_image(chunk_n, frame_idx)

