# coding=utf-8
from __future__ import print_function, division, absolute_import

import os.path
import warnings
import operator
import time
import logging
import abc
import numpy as np

from imgstore.constants import DEVNULL, SQLITE3_INDEX_FILE, STORE_MD_FILENAME, STORE_LOCK_FILENAME, STORE_MD_KEY, \
    STORE_INDEX_FILENAME, EXTRA_DATA_FILE_EXTENSIONS, FRAME_MD as _FRAME_MD
from imgstore.util import ImageCodecProcessor, JsonCustomEncoder, motif_extra_data_h5_to_df, motif_extra_data_json_to_df, motif_extra_data_h5_attrs
from imgstore.index import ImgStoreIndex
from imgstore.stores.utils.logging import _Log
from imgstore.stores.utils.path import get_fullpath
from imgstore.stores.utils.datetime import parse_old_time, parse_new_time
from imgstore.stores.base.write import WritingStore
from imgstore.stores.base.read import ReadingStore

_VERBOSE_VERY = False  # overrides the other and prints all logs to stdout

from imgstore.stores.utils.mixins import MIXINS

logger = logging.getLogger(__name__)

# note: frame_idx always refers to the frame_idx within the chunk
# whereas frame_index refers to the global frame_index from (0, frame_count]

class AbstractStore(abc.ABC):

    @classmethod
    def supported_formats(cls):
        raise NotImplementedError

    def _save_image(self, img, frame_number, frame_time):  # pragma: no cover
        raise NotImplementedError

    def _save_chunk(self, old, new):  # pragma: no cover
        raise NotImplementedError

    def _load_image(self, idx):  # pragma: no cover
        raise NotImplementedError

    def _load_chunk(self, n):  # pragma: no cover
        raise NotImplementedError

    def _find_chunks(self, chunk_numbers):  # pragma: no cover
        raise NotImplementedError

class AbstractImgStore(AbstractStore):

    @property
    def created(self):
        return self._created_utc, self._timezone_local

    @property
    def uuid(self):
        return self._uuid

    @property
    def chunks(self):
        """ the number of non-empty chunks that contain images """
        if self._mode == 'r':
            return list(self._index.chunks)
        else:
            return list(range(0, self._chunk_n))

    @property
    def user_metadata(self):
        return self._user_metadata

    @property
    def filename(self):
        return self._basedir

    @property
    def full_path(self):
        return os.path.join(self._basedir, STORE_MD_FILENAME)

    @property
    def image_shape(self):
        # if encoding is specified, we always decode to bgr (color)
        if self._encoding:
            return self._imgshape[0], self._imgshape[1], 3
        else:
            return self._imgshape

    @property
    def duration(self):
        if np.isreal(self._tN) and np.isreal(self._t0):
            return self._tN - self._t0
        return 0.

    @property
    def mode(self):
        return self._mode

    @staticmethod
    def _extract_only_frame(basedir, chunk_n, frame_n, smd):
        raise NotImplementedError

    def __len__(self):
        return self.frame_count


    def _iter_chunk_n_and_chunk_paths(self):
        if self._chunk_n_and_chunk_paths is None:
            t0 = time.time()
            self._chunk_n_and_chunk_paths = self._find_chunks(chunk_numbers=None)
            self._log.debug('found %s chunks in in %fs' % (len(self._chunk_n_and_chunk_paths), time.time() - t0))

        for cn, cp in sorted(self._chunk_n_and_chunk_paths, key=operator.itemgetter(0)):
            yield cn, cp



class _ImgStore(AbstractImgStore, ReadingStore, WritingStore, *MIXINS):
    _version = 2
    _supported_modes = ''

    FRAME_MD = _FRAME_MD

    # noinspection PyShadowingBuiltins
    def __init__(self, basedir, mode, fps=25, imgshape=None, imgdtype=np.uint8, chunksize=None, metadata=None,
                 encoding=None, write_encode_encoding=None, format=None, index=None, **kwargs):
        if mode not in self._supported_modes:
            raise ValueError('mode not supported')

        if imgdtype is not None:
            # ensure this is a string
            imgdtype = np.dtype(imgdtype).name

        self._basedir = basedir
        self._fps = fps
        self._mode = mode
        self._imgshape = ()
        self._imgdtype = ''
        self._chunksize = 0
        self._encoding = None
        self._format = None
        self._codec_proc = ImageCodecProcessor()  # used in read and write mode
        self._decode_image = None
        self._encode_image = None
        self._uuid = None

        self._metadata = {"framerate": fps, **kwargs}
        self._user_metadata = {}
        self._frame_metadata = {}
        self._write_imgshape = ()

        self._tN = self._t0 = time.time()

        self._created_utc = self._timezone_local = None

        self.frame_min = np.nan
        self.frame_max = np.nan
        self.frame_number = np.nan
        self.frame_count = 0
        self.frame_time = np.nan


        self._log = logging.getLogger('imgstore')
        if _VERBOSE_VERY:
            _VERBOSE_DEBUG_GETS = VERBOSE_DEBUG_CHUNKS = True
            self._log = _Log

        self._chunk_n = 0
        self._chunk_current_frame_idx = -1
        self._chunk_n_and_chunk_paths = None
        self._last_img = None

        # file pointer and filename of a file which can be used to store additional data per frame
        # (this is only created if data is actually stored)
        self._extra_data_fp = self._extra_data_fn = None

        if mode == 'w':
            if None in (imgshape, imgdtype, chunksize, format):
                raise ValueError('imgshape, imgdtype, chunksize, format must not be None')
            self._frame_n = 0
            self._init_write(imgshape, imgdtype, chunksize, metadata, encoding, write_encode_encoding, format)
        elif mode == 'r':
            self._init_read()
            self._index = None
            if index is None:
                # as we need to read the chunks, populate the chunk cache now too (could be an expensive operation
                # if there are thousands of chunks)
                t0 = time.time()
                self._chunk_n_and_chunk_paths = self._find_chunks(chunk_numbers=None)
                self._log.debug('found %s chunks in in %fs' % (len(self._chunk_n_and_chunk_paths), time.time() - t0))

                if self.index_db_exists:
                    # NOTE:
                    # This is a new feature that just performs a db connection
                    # to a sqlite3 version of the npz files of the imgstore
                    # (same info but in a single file)
                    # instead of regenerating a sqlite3 db every time
                    self._index = ImgStoreIndex.new_from_file(self.index_db_path)
                else:
                    self._index = ImgStoreIndex.new_from_chunks(self._chunk_n_and_chunk_paths)
            elif (index is not None) and isinstance(index, ImgStoreIndex):
                self._log.debug('using supplied index')
                self._index = index
            else:
                raise TypeError('index must be of type ImgStoreIndex')

            self._t0 = self._index.frame_time_min
            self._tN = self._index.frame_time_max
            self.frame_count = self._index.frame_count
            self.frame_min = self._index.frame_min
            self.frame_max = self._index.frame_max

            # reset to the start of the file and load the first chunk
            self._load_chunk(0)
            assert self._chunk_current_frame_idx == -1
            assert self._chunk_n == 0
            self.frame_number = np.nan  # we haven't read any frames yet


    def _calculate_image_shape(self, imgshape, fmt):
        _imgshape = list(imgshape)
        # bitwise and with -2 truncates downwards to even
        _imgshape[0] = int(_imgshape[0]) & -2
        _imgshape[1] = int(_imgshape[1]) & -2
        return tuple(_imgshape)

    def get_chunk(self, chunk):
        """
        Place the store so the next frame is the first frame of the chunk
        """
        self._log.debug(f"{self}.get_chunk({chunk})")
        assert chunk is not None
        first_fn=self._index.get_chunk_metadata(chunk)["frame_number"][0]
        img, (frame_number, frame_time) = self.get_image(max(0, first_fn))
        return img, (frame_number, frame_time)
    
    @property
    def frame_metadata(self):
        if len(self._frame_metadata) == 0:
            self._frame_metadata=self._index.get_all_metadata()
        return self._frame_metadata

    @property
    def index_db_path(self):
        return os.path.join(self._basedir, SQLITE3_INDEX_FILE)

    @property
    def index_db_exists(self):
        return os.path.exists(self.index_db_path)

    def frame_number2frame_index(self, frame_number, chunk=None):
        if chunk:
            try:
                return self._get_chunk_metadata(chunk)["frame_number"].index(frame_number)

            except ValueError:
                warnings.warn(f"{frame_number} not found in {self}-{chunk}")
                return

        else:
            return self._index.find_chunk("frame_number", frame_number)[1]


    @classmethod
    def supports_format(cls, fmt):
        return fmt in cls.supported_formats()

    def _get_chunk_metadata(self, chunk_n):
        return self._index.get_chunk_metadata(chunk_n)

    def get_frame_metadata(self):
        return self._index.get_all_metadata()

    def disable_decoding(self):
        self._decode_image = lambda x: x


    def close(self, save_index=False):
        if self._mode in 'wa':
            self._save_chunk(self._chunk_n, None)
            # noinspection PyBroadException
            try:
                if os.path.isfile(os.path.join(self._basedir, STORE_LOCK_FILENAME)):
                    os.remove(os.path.join(self._basedir, STORE_LOCK_FILENAME))
            except OSError:
                # noinspection PyArgumentList
                self._log.warn('could not remove lock file', exc_info=True)
            except Exception:
                # noinspection PyArgumentList
                self._log.warn('could not remove lock file (unknown error)', exc_info=True)

            if save_index:
                index = ImgStoreIndex.new_from_chunks(self._find_chunks(chunk_numbers=None))
                index.to_file(os.path.join(self._basedir, STORE_INDEX_FILENAME))


    def empty(self):
        if self._mode != 'w':
            raise ValueError('can only empty stores for writing')

        self.close()

        self._tN = self._t0 = time.time()

        self.frame_min = np.nan
        self.frame_max = np.nan
        self.frame_number = np.nan
        self.frame_count = 0
        self.frame_time = np.nan

        self._chunk_n = 0
        self._chunk_n_and_chunk_paths = None

        if self._extra_data_fp is not None:
            self._extra_data_fp.close()
            os.unlink(self._extra_data_fn)
            self._extra_data_fp = self._extra_data_fn = None

        self._frame_n = 0

__all__ = [_ImgStore]
