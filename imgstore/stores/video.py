import logging
import shutil
import os.path
import glob
import operator
import warnings

import codetiming
import cv2
import numpy as np
from imgstore.constants import STORE_MD_KEY, VERBOSE_DEBUG_CHUNKS
from imgstore.util import ensure_color, ensure_grayscale
from imgstore.stores.utils.formats import get_formats
from imgstore.stores.base import _ImgStore

try:
    import cv2cuda # type: ignore
    CV2CUDA_AVAILABLE=True
except Exception:
    cv2cuda=None
    CV2CUDA_AVAILABLE=False

isColor=False
ONLY_ALLOW_EVEN_SIDES=True

logger = logging.getLogger(__name__)

def find_chunks_video(basedir, ext, chunk_numbers=None):
    if chunk_numbers is None:
        avis = map(os.path.basename, glob.glob(os.path.join(basedir, '*%s' % ext)))
        chunk_numbers = list(map(int, map(operator.itemgetter(0), map(os.path.splitext, avis))))
    data = list(zip(chunk_numbers, tuple(os.path.join(basedir, '%06d' % n) for n in chunk_numbers)))
    return data

class VideoImgStore(_ImgStore):
    _supported_modes = 'wr'
    _cv2_fmts = get_formats(cache=True, video=True)

    _DEFAULT_CHUNKSIZE = 10000

    def __init__(self, **kwargs):

        self._cap = None
        self._cap_hq = None
        self._capfn = None
        self._capfn_hq = None
        self._last_capfn=None


        fmt = kwargs.get('format')
        # backwards compat
        if fmt == 'mjpeg':
            kwargs['format'] = fmt = 'mjpeg/avi'

        # default to seeking enable
        seek = kwargs.pop('seek', True)
        # keep compat with VideoImgStoreFFMPEG
        kwargs.pop('gpu_id', None)

        if kwargs['mode'] == 'w':
            imgshape = kwargs['imgshape']

            if 'chunksize' not in kwargs:
                kwargs['chunksize'] = self._DEFAULT_CHUNKSIZE

            try:
                self._codec = self._cv2_fmts[fmt]
            except KeyError:
                raise ValueError('only %r supported', (self._cv2_fmts.keys(),))

            self._color = (imgshape[-1] == 3) & (len(imgshape) == 3)

            metadata = kwargs.get('metadata', {})
            try:
                metadata[STORE_MD_KEY] = {'extension': '.%s' % fmt.split('/')[1]}
            except Exception as error:
                warnings.warn(f"{fmt} is not splittable", stacklevel=2)
                raise error
                
            kwargs['metadata'] = metadata
            kwargs['encoding'] = kwargs.pop('encoding', None)

        _ImgStore.__init__(self, **kwargs)

        self._supports_seeking = seek
        if self._supports_seeking:
            self._log.info('seeking enabled on store')
        else:
            self._log.info('seeking NOT enabled on store (will fallback to sequential reading)')

        if self._mode == 'r':
            if self.supports_format(self._format) or \
                    ((self._metadata.get('class') == 'VideoImgStoreFFMPEG') and
                     (('264' in self._format) or ('nvenc-' in self._format))):
                check_imgshape = tuple(self._imgshape)
                #check_imgshape = self._calculate_image_shape(self._imgshape, '')

            else:
                check_imgshape = tuple(self._imgshape)

            if check_imgshape != self._imgshape:
                self._log.warn('previous store had incorrect image_shape: corrected %r -> %r' % (
                    self._imgshape, check_imgshape))
                self._imgshape = check_imgshape
            self._write_imgshape = check_imgshape
            self._color = (self._imgshape[-1] == 3) & (len(self._imgshape) == 3)

        self._log.info("store is native color: %s (or grayscale with encoding: '%s')" % (self._color, self._encoding))

    def _readability_check(self, smd_class, smd_version):
        can_read = {'VideoImgStoreFFMPEG', 'VideoImgStore',
                    getattr(self, 'class_name', self.__class__.__name__)}
        if smd_class not in can_read:
            raise ValueError('incompatible store, can_read:%r opened:%r' % (can_read, smd_class))
        if smd_version != self._version:
            raise ValueError('incompatible store version')
        return True

    @staticmethod
    def _get_chunk_extension(metadata):
        # forward compatibility
        try:
            return metadata['extension']
        except KeyError:
            # backward compatibility with old mjpeg stores
            if metadata['format'] == 'mjpeg':
                return '.avi'
            # backwards compatibility with old bview/motif stores
            return '.mp4'

    @property
    def _ext(self):
        return self._get_chunk_extension(self._metadata)

    @property
    def _chunk_paths(self):
        ext = self._ext
        return ['%s%s' % (p[1], ext) for p in self._chunk_n_and_chunk_paths]

    def _find_chunks(self, chunk_numbers):
        return find_chunks_video(self._basedir, self._ext, chunk_numbers)

    def _save_image(self, img, frame_number, frame_time):
        # we always write color because its more supported
            
        if self._color:
            frame = ensure_color(img)
        else:
            frame = ensure_grayscale(img)

        self._cap.write(frame)
        if self._chunk_current_frame_idx > 0 and not os.path.isfile(self._capfn):
            raise Exception(
                f"""
                {self._capfn} could not be created.
                Probably, your opencv build does support writing this codec ({self._codec})
                """
            )

        self._save_image_metadata(frame_number, frame_time)

    def _save_chunk(self, old, new):
        if self._cap is not None:
            self._cap.release()
            if self._cap_hq is not None: self._cap_hq.release()
            self._save_chunk_metadata(os.path.join(self._basedir, '%06d' % old))

        if new is not None:
            fn = os.path.join(self._basedir, '%06d%s' % (new, self._ext))
            h, w = self._imgshape[:2]
            try:
                
                if self._codec == "h264_nvenc" and not CV2CUDA_AVAILABLE:
                    self._codec=self._cv2_fmts['avc1/mp4']


                if self._codec == "h264_nvenc" and new != 0:
                    self._cap = cv2cuda.VideoWriter(
                        filename=fn,
                        apiPreference="FFMPEG",
                        fourcc="h264_nvenc",
                        fps=self.fps,
                        frameSize=(w, h),
                        isColor=self._color
                    )
            
                else:
                    if new == 0 and self._codec == "h264_nvenc":
                        codec = cv2.VideoWriter_fourcc(*"DIVX")
                        filename = fn.replace(".mp4", ".avi")
                    else:
                        codec = self._codec
                        filename = fn

                    self._cap = cv2.VideoWriter(
                        filename=filename,
                        apiPreference=cv2.CAP_FFMPEG,
                        fourcc=codec,
                        fps=self.fps,
                        frameSize=(w, h),
                        isColor=self._color
                    )

            except TypeError as error:
                self._log.error(
                    f"""
                    {error}
                    old (< 3.2) cv2 not supported (this is {cv2.__version__})
                    filename: {fn},
                    fourcc: {self._codec},
                    frameSize: {(w, h)},
                    isColor: {self._color}
                    """
                )
                self._cap = cv2.VideoWriter(
                    filename=fn,
                    fourcc=self._codec,
                    fps=self.fps,
                    frameSize=(w, h),
                    isColor=self._color
                )

            self._capfn = fn
            self._new_chunk_metadata(os.path.join(self._basedir, '%06d' % new))
            self.frame_idx = 0


    @property
    def burn_in_period(self):
        pass
        if self._capfn.endswith(".mp4") and os.path.exists(self._capfn_hq) and self._cap_hq is not None:
            return self.fps * 5
        else:
            return -1

    @staticmethod
    def _read(cap):
        ret, img = cap.read()
        if not ret:
            return ret, None
        if ONLY_ALLOW_EVEN_SIDES:
            dims = list(img.shape)
            for i, dimension in enumerate(img.shape):
                if dimension % 2 != 0:
                    dims[i] -= 1
            
            img = img[:dims[0], :dims[1]].copy(order="C")
        return ret, img               



    def _load_image(self, idx):


        with codetiming.Timer(text="Reading image took {milliseconds:.0f} ms", logger=logger.debug):

            if idx < self.burn_in_period:
                cap  = self._cap_hq
            else:
                cap = self._cap
            
            if (idx - self._chunk_current_frame_idx) == 0 and self._last_img is not None:
                ret = True
                _img = self._last_img.copy()

            else:
                if self._supports_seeking:
                    # only seek if we have to, otherwise take the fast path
                    if (idx - self._chunk_current_frame_idx) != 1:
                        # print(f"{idx - self._chunk_current_frame_idx} frames in the future")
                        cap.set(getattr(cv2, "CAP_PROP_POS_FRAMES", 1), idx)
                else:
                    raise Exception("Only videos with seeking support enabled")
                    # TODO
                    # Return back support for videos without seeking
                    # support by dealing here both with _cap and _cap_hq 
                    # if idx <= self._chunk_current_frame_idx:
                    #     self._load_chunk(self._chunk_n, _force=True)

                    # i = self._chunk_current_frame_idx + 1
                    # while i < idx:
                    #     _, img = cap.read()
                    #     i += 1

                ret, _img = self._read(cap)
                if not ret:
                    return None, (None, None)
                if cap is self._cap_hq:
                    self._cap.set(getattr(cv2, "CAP_PROP_POS_FRAMES", 1), idx+1)

            assert ret, f"Cannot read frame from {self._capfn}"

        with codetiming.Timer(text="Ensuring color or grayscale took {milliseconds:.0f} ms", logger=logger.debug):
            if self._color:
                # almost certainly no-op as opencv usually returns color frames....
                img = ensure_color(_img)
            else:
                img = ensure_grayscale(_img)

        if self._metadata.get("apply_blur", False):
            with codetiming.Timer(text="Applying gaussian blur took {milliseconds:.0f} ms", logger=logger.debug):
                # print("Applying gaussian blur")
                img = cv2.GaussianBlur(img, (0, 0), self._metadata["apply_blur"])

        with codetiming.Timer(text="Copying image took {milliseconds:.0f} ms", logger=logger.debug):
            self._last_img = img.copy()

        return img, (self._chunk_md['frame_number'][idx], self._chunk_md['frame_time'][idx])

    def _load_chunk(self, n, _force=False):
        fn = os.path.join(self._basedir, '%06d%s' % (n, self._ext))
        if _force or (fn != self._capfn):
            if self._cap is not None:
                self._cap.release()
            if self._cap_hq is not None:
                self._cap_hq.release()

            self._log.debug('loading chunk %s' % n)
            self._capfn = fn

            caps = []
            fns = []

            if os.path.exists(self._capfn):
                self._cap = cv2.VideoCapture(self._capfn)
                caps.append(self._cap)
                fns.append(self._capfn)

            self._capfn_hq = os.path.splitext(fn)[0] + ".avi"
            self._cap_hq = cv2.VideoCapture(self._capfn_hq)
            caps.append(self._cap_hq)
            fns.append(self._capfn_hq)

            self._chunk_current_frame_idx = -1

            for fn, cap in zip(fns, caps):
                if not cap.isOpened():
                    if fn == self._capfn_hq:
                        self._log.warning(f"Cannot open {fn}")
                        self._cap_hq = None
                    else:
                        raise Exception("OpenCV unable to open %s" % fn)

            self._chunk_md = self._index.get_chunk_metadata(n)

        self._chunk_n = n

    @classmethod
    def supported_formats(cls):
        # remove the duplicate
        fmts = list(cls._cv2_fmts.keys())
        return fmts

    @classmethod
    def supports_format(cls, fmt):
        return fmt in cls._cv2_fmts

    def _extract_only_frame(self, basedir, chunk_n, frame_n, smd):
        capfn = os.path.join(basedir, '%06d%s' % (chunk_n,
                                                  VideoImgStore._get_chunk_extension(smd)))
        # noinspection PyArgumentList
        cap = cv2.VideoCapture(capfn)

        if VERBOSE_DEBUG_CHUNKS:
            log = logging.getLogger('imgstore')
            log.debug('opening %s chunk %d frame_idx %d' % (capfn, chunk_n, frame_n))

        try:
            if frame_n > 0:
                cap.set(getattr(cv2, "CAP_PROP_POS_FRAMES", 1), frame_n)
            _, img = self._read(cap)
            return img
        finally:
            cap.release()

    @property
    def lossless(self):
        return False

    def close(self, **kwargs):
        super(VideoImgStore, self).close(**kwargs)
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            self._last_capfn=self._capfn
            self._capfn = None
        if self._cap_hq is not None:
            self._cap_hq.release()
            self._cap_fn = None

    def empty(self):
        _ImgStore.empty(self)
        # use find_chunks because it removes also invalid chunks
        for _, chunk_path in self._find_chunks(chunk_numbers=None):
            os.unlink(chunk_path + self._ext)
            self._remove_index(chunk_path)

    def insert_chunk(self, video_path, frame_numbers, frame_times, move=True):
        assert len(frame_numbers) == len(frame_times)
        assert video_path.endswith(self._ext)

        self._new_chunk_metadata(os.path.join(self._basedir, '%06d' % self._chunk_n))
        self._chunk_md['frame_number'] = np.asarray(frame_numbers)
        self._chunk_md['frame_time'] = np.asarray(frame_times)
        self._save_chunk_metadata(os.path.join(self._basedir, '%06d' % self._chunk_n))

        vid = os.path.join(self._basedir, '%06d%s' % (self._chunk_n, self._ext))
        if move:
            shutil.move(video_path, vid)
        else:
            shutil.copy(video_path, vid)

        self._chunk_n += 1
