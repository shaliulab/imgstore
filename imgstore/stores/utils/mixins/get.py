_VERBOSE_DEBUG_GETS = False

from imgstore.constants import VERBOSE_DEBUG_CHUNKS

class GetMixin:

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
                raise EOFError

            self._load_chunk(next_chunk)

            # first frame is start of chunk
            idx = 0
            frame_number = self._chunk_md['frame_number'][idx]

        return frame_number, idx


    def _get_image(self, chunk_n, frame_idx):
        if chunk_n is not None:
            self._load_chunk(chunk_n)

        # ensure the read works before setting frame_number
        _img, (_frame_number, _frame_timestamp) = self._load_image(frame_idx)
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

