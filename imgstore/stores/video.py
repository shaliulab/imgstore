import logging
import shutil
import os.path
import glob
import operator
import tqdm
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

def quality_control(metadata, cap):
    """
    Verify integrity of file being read
    """

    diffs = {
        "height": abs(metadata["imgshape"][0] - cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "width": abs(metadata["imgshape"][1] - cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "duration": metadata["chunksize"] - cap.get(cv2.CAP_PROP_FRAME_COUNT),
    }

    for diff in diffs:
        assert diffs[diff] <= 1, f"""
        {diff} is > 1 ({diffs[diff]})
        {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}
        If either of these values is 0, the video is corrupted, maybe during upload from the lab to the vsc,
        try sending it again
        """
        if diffs[diff] == 1:
            warnings.warn(f"{diff} difference is 1")

    testing_points = [0.1, 0.5, 0.9]
    for point in testing_points:
        pos = int(point * metadata["chunksize"])
        cap.set(1, pos)
        assert cap.get(1) == pos

    cap.set(1, 0)


def find_chunks_video(basedir, ext, chunk_numbers=None):
    if chunk_numbers is None:
        avis = map(os.path.basename, glob.glob(os.path.join(basedir, '*%s' % ext)))
        avis = [e for e in avis if e.count(".")==1]
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

        self._min_bitrate = kwargs.pop("min_bitrate", None)
        self._max_bitrate = kwargs.pop("max_bitrate", None)
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

    # def _readability_check(self, smd_class, smd_version):
    #     can_read = {'VideoImgStoreFFMPEG', 'VideoImgStore',
    #                 getattr(self, 'class_name', self.__class__.__name__)}
    #     if smd_class not in can_read:
    #         raise ValueError('incompatible store, can_read:%r opened:%r' % (can_read, smd_class))
    #     if smd_version != self._version:
    #         raise ValueError('incompatible store version')
    #     return True

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

        # print(self.frame_idx, self.burnin_period)
        if self.in_burnin_period:
            # print(f"Writing frame {frame.shape} in burn in period {self.burnin_period}")
            self._cap_hq.write(frame)

        self._cap.write(frame)
        if self._chunk_current_frame_idx > 0 and not os.path.isfile(self._capfn):
            raise Exception(
                f"""
                {self._capfn} could not be created.
                Probably, your opencv build does support writing this codec ({self._codec})
                """
            )

        self._save_image_metadata(frame_number, frame_time)



    def _finish_chunk(self, old):
        if self._cap is not None:
            self._cap.release()
            if self._cap_hq is not None: self._cap_hq.release()
            filename=os.path.join(self._basedir, '%06d' % old)
            print("SAVE CHUNK METADATA", filename)
            self._save_chunk_metadata(filename)

    def _start_chunk(self, old, new):
        """
        Update

        self._capfn
        self._cap

        self._capfn_hq
        self._cap_hq
        """
        fn = os.path.join(self._basedir, '%06d%s' % (new, self._ext))
        h, w = self._imgshape[:2]

        # if self.burnin_period > 0:
        self._capfn_ = fn
        self._capfn_hq_ = self._capfn_.replace(".mp4", ".avi")
        self._cap_hq_ = cv2.VideoWriter(
                filename=self._capfn_hq_,
                fourcc=cv2.VideoWriter_fourcc(*"DIVX"),
                fps=self.fps,
                frameSize=(w, h),
                isColor=self._color
            )
        try:

            if self._codec == "h264_nvenc" and not CV2CUDA_AVAILABLE:
                self._codec=self._cv2_fmts['avc1/mp4']


            if self._codec == "h264_nvenc" and (new != 0 or (old is not None)):
                print(f" --> {fn}")
                self._cap_ = cv2cuda.VideoWriter(
                    filename=self._capfn_,
                    apiPreference="FFMPEG",
                    fourcc="h264_nvenc",
                    fps=self.fps,
                    frameSize=(w, h),
                    isColor=self._color,
                    maxframes=self._chunksize,
                    **self._metadata["encoder_kwargs"]
                )

            else:
                if new == 0 and self._codec == "h264_nvenc":
                    codec = cv2.VideoWriter_fourcc(*"DIVX")
                    filename = self._capfn_.replace(".mp4", ".avi")
                else:
                    codec = self._codec
                    filename = self._capfn_

                self._cap_ = cv2.VideoWriter(
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
            self._cap_ = cv2.VideoWriter(
                filename=self._capfn_,
                fourcc=self._codec,
                fps=self.fps,
                frameSize=(w, h),
                isColor=self._color
            )



    def _save_chunk(self, old, new):
        print(f"Calling _save_chunk. old={old}, new={new}")
        self._finish_chunk(old)

        if new is not None:
            self._start_chunk(old, new)

            self._capfn = self._capfn_
            self._capfn_hq = self._capfn_hq_
            self._cap = self._cap_
            self._cap_hq = self._cap_hq_

            self._new_chunk_metadata(os.path.join(self._basedir, '%06d' % new))
            self.frame_idx = 0
            return self._capfn

    @property
    def burnin_period(self):
        if self._ext == ".mp4" and os.path.exists(self._capfn_hq) and self._cap_hq is not None:
            if self._metadata.get("burnin_period", 2)  == 0:
                period = 0 # s
            else:
                period = 2
            return self.fps * period
        else:
            return -1


    @property
    def in_burnin_period(self):
        in_burnin_period_status=self.frame_idx < self.burnin_period
        return in_burnin_period_status


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

    def frame_is_miscoded(self, frame_number):
        return False
        frame_number_criterion = frame_number % 250 < 2
        if not frame_number_criterion:
            return False

        if self._mode == "r":
            codec_criterion = self._ext == ".mp4"
        elif self._mode == "w":
            codec_criterion = self._codec == "h264_nvenc"

        return frame_number_criterion and codec_criterion

    def _load_image(self, idx):
        try:
            with open("debug_imgstore.txt", "r") as filehandle:
                message = filehandle.read().strip("\n")
        except:
            message = "no-debug"

        with codetiming.Timer(text="Reading image took {milliseconds:.0f} ms", logger=logger.debug):
            if (idx+1) < self.burnin_period:
                cap  = self._cap_hq
                if message == "debug": print("Loading from high quality capture")
            else:
                cap = self._cap
                if message == "debug": print("Loading from normal capture")


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
                if not ret and idx < cap.get(7):
                    warnings.warn("Weird behavior of VideoCapture object. Resetting")
                    cap.set(1, 0)
                    for _ in tqdm.tqdm(range(idx)):
                        ret, _  = cap.read()
                        assert ret
                    # del cap
                    # cap = cv2.VideoCapture(self._capfn)
                    # cap.set(getattr(cv2, "CAP_PROP_POS_FRAMES", 1), idx)
                    # self._cap = cap
                    ret, _img = self._read(cap)

                if self.frame_is_miscoded(idx):
                    ret_miscoded, img_miscoded = self._read_miscoded_frame(self._chunk_n, self._chunk_md['frame_number'][idx])
                    if ret_miscoded:
                        _img = img_miscoded
                        ret = ret_miscoded

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
                logger.debug(f"Applying blur {self._metadata['apply_blur']}")
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
                if n > 1:
                    quality_control(self._metadata, self._cap)

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
            self._capfn = None

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
