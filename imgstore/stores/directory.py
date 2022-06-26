import itertools
import os.path

import cv2
import numpy as np

try:
    import bloscpack # type: ignore
except ImportError:
    bloscpack = None


from imgstore.util import ensure_color, ensure_grayscale
from imgstore.stores.base import _ImgStore
from imgstore.stores.utils.formats import get_formats

class DirectoryImgStore(_ImgStore):
    _supported_modes = 'wr'

    _cv2_fmts = get_formats(cache=True, directory=True)
    _raw_fmts = get_formats(cache=True, raw=True)

    _DEFAULT_CHUNKSIZE = 200

    def __init__(self, **kwargs):

        self._chunk_cdir = ''
        self._chunk_md = {}

        # keep compat with VideoImgStore
        kwargs.pop('videosize', None)
        # keep compat with VideoImgStoreFFMPEG
        kwargs.pop('seek', None)
        kwargs.pop('gpu_id', None)

        if kwargs['mode'] == 'w':
            if 'chunksize' not in kwargs:
                kwargs['chunksize'] = self._DEFAULT_CHUNKSIZE
            kwargs['encoding'] = kwargs.pop('encoding', None)

        _ImgStore.__init__(self, **kwargs)

        self._color = (self._imgshape[-1] == 3) & (len(self._imgshape) == 3)

        if (self._mode == 'w') and (self._format == 'pgm') and self._color:
            self._log.warn("store created with color image shape but using grayscale 'pgm' format")

        if self._format not in itertools.chain(self._cv2_fmts, ('npy', 'bpk')):
            raise ValueError('unknown format %s' % self._format)

        if (self._format == 'bpk') and (bloscpack is None):
            raise ValueError('bloscpack not installed or available')

        if (self._format == 'npy') and (np.__version__ < '1.9.0') and (self._mode in 'wa'):
            # writing to npy takes an unecessary copy in memory [1], which was fixed in version 1.9.0
            # [1] https://www.youtube.com/watch?v=TZdqeEd7iTM
            pass

    def _save_image(self, img, frame_number, frame_time):
        dest = os.path.join(self._chunk_cdir, '%06d.%s' % (self._frame_n % self._chunksize, self._format))

        if self._format in self._cv2_fmts:
            if self._format == 'ppm':
                img = ensure_color(img)
            elif self._format == 'pgm':
                img = ensure_grayscale(img)
            cv2.imwrite(dest, img)
        elif self._format == 'npy':
            np.save(dest, img)
        elif self._format == 'bpk':
            bloscpack.pack_ndarray_file(img, dest)

        self._save_image_metadata(frame_number, frame_time)

    def _save_chunk(self, old, new):
        if old is not None:
            self._save_chunk_metadata(os.path.join(self._chunk_cdir, 'index'))
        if new is not None:
            self._chunk_cdir = os.path.join(self._basedir, '%06d' % new)
            os.mkdir(self._chunk_cdir)
            self._new_chunk_metadata(os.path.join(self._basedir, '%06d' % new, 'index'))

    def _find_chunks(self, chunk_numbers):
        if chunk_numbers is None:
            immediate_dirs = next(os.walk(self._basedir))[1]
            chunk_numbers = list(map(int, immediate_dirs))  # N.B. need list, as we iterate it twice
        return list(zip(chunk_numbers, tuple(os.path.join(self._basedir, '%06d' % int(n), 'index')
                                             for n in chunk_numbers)))

    # noinspection PyShadowingBuiltins
    @staticmethod
    def _open_image(path, format, color):
        if format in DirectoryImgStore._cv2_fmts:
            flags = cv2.IMREAD_COLOR if color else cv2.IMREAD_GRAYSCALE
            img = cv2.imread(path, flags)
        elif format == 'npy':
            img = np.load(path)
        elif format == 'bpk':
            with open(path, 'rb') as reader:
                img = bloscpack.numpy_io.unpack_ndarray(bloscpack.file_io.CompressedFPSource(reader))
        else:
            # Won't get here unless we relax checks in constructor, but better safe
            raise ValueError('unknown format %s' % format)
        return img

    def _load_image(self, idx):
        path = os.path.join(self._chunk_cdir, '%06d.%s' % (idx, self._format))
        img = self._open_image(path, self._format, self._color)
        self._last_img = img.copy()
        return img, (self._chunk_md['frame_number'][idx], self._chunk_md['frame_time'][idx])

    def _load_chunk(self, n):
        cdir = os.path.join(self._basedir, '%06d' % n)
        if cdir != self._chunk_cdir:
            self._log.debug('loading chunk %s' % n)
            self._chunk_cdir = cdir
            self._chunk_md = self._index.get_chunk_metadata(n)

        self._chunk_n = n
        self._chunk_current_frame_idx = -1  # not used in DirectoryImgStore, but maintain compat

    @staticmethod
    def _extract_only_frame(basedir, chunk_n, frame_n, smd):
        fmt = smd['format']
        cdir = os.path.join(basedir, '%06d' % chunk_n)
        path = os.path.join(cdir, '%06d.%s' % (frame_n, fmt))

        imgshape = tuple(smd['imgshape'])
        color = (imgshape[-1] == 3) & (len(imgshape) == 3)

        return DirectoryImgStore._open_image(path, fmt, color)

    @classmethod
    def supported_formats(cls):
        fmts = list(cls._cv2_fmts) + list(cls._raw_fmts)
        if bloscpack is None:
            fmts.remove('bpk')
        return fmts

    @property
    def lossless(self):
        return self._format != 'jpg'

