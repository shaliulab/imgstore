import abc
import os.path
import datetime
import pytz
import tzlocal
import uuid
import cv2
import yaml
import numpy as np

from imgstore.constants import STORE_MD_FILENAME, STORE_LOCK_FILENAME, STORE_MD_KEY, STORE_INDEX_FILENAME


def annotate_frame(frame, metadata):
    
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 0.5
    FONT_THICKNESS = 2
    if len(frame.shape) == 2:
        n_channels = 1
    else:
        n_channels = frame.shape[2]

    bg_color = (255, ) * n_channels
    label_color = (0, ) * n_channels

    label = "\n".join([f"{k}: {v}" for k, v in metadata.items()])
    
    (label_width, label_height), baseline = cv2.getTextSize(label, FONT, FONT_SCALE, FONT_THICKNESS)
    if n_channels != 1:
        label_patch = np.zeros((label_height + baseline, label_width, n_channels), np.uint8)
    else:
        label_patch = np.zeros((label_height + baseline, label_width), np.uint8)
        
    label_patch[:,:] = bg_color
    cv2.putText(label_patch, label, (0, label_height), FONT, FONT_SCALE, label_color, FONT_THICKNESS)
    
    if n_channels != 1:
        frame[:(label_height+baseline), :label_width, :] = label_patch
    else:
        frame[:(label_height+baseline), :label_width] = label_patch
    return frame

alpha=7
beta=20
gamma=4

lookUpTable = np.empty((1,256), np.uint8)
for i in range(256):
    lookUpTable[0,i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)

alpha=10
beta=10
gamma=10
lookUpTable = np.empty((1,256), np.uint8)
for i in range(256):
    lookUpTable[0,i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)

kernel=np.ones((3, 3))
ratio=25
zero=20
bias=22

def process_image(img, threshold=None):
    # img = cv2.imread("/Users/Antonio/FSLLab/Projects/FlyHostel4/notebooks/2022-10-23_contrast/18518318-2022-10-23-213232.png", cv2.IMREAD_GRAYSCALE)
    new_image = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    res = cv2.LUT(new_image, lookUpTable)
    blur=cv2.GaussianBlur(res,(5, 5), 1)

    # _, thr = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY_INV)
    # close = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel)
    # erosion = 255-cv2.erode(close, kernel, 2)
    out = blur
    return out

class WritingStore(abc.ABC):


    def add_image(self, img, frame_number, frame_time, annotate=True, start_next_chunk=True):

        if img.shape != self._write_imgshape:
            img = img[:self._write_imgshape[0], :self._write_imgshape[1]].copy(order="C")

        # if "lowres" in self._basedir:
        #     # mean = img.mean()
        #     # print(mean)
        #     # threshold=bias+(ratio*(mean-zero))

        #     # print(threshold)
        #     img = process_image(img, threshold=None)

        new_fn = None

        if annotate:
            img = annotate_frame(img, metadata={"fn": self.frame_idx-1})

        self._save_image(self._encode_image(img), frame_number, frame_time)

        self.frame_max = np.nanmax((frame_number, self.frame_max))
        self.frame_min = np.nanmin((frame_number, self.frame_min))
        self.frame_number = frame_number
        self.frame_time = frame_time

        if self._frame_n == 0:
            self._t0 = frame_time
        self._tN = frame_time

        self._frame_n += 1

        
        if self.frame_is_miscoded(self._frame_n):
            self._save_miscoded_frame(img, self._chunk_n, self.frame_idx)


        if (self._frame_n % self._chunksize) > 0.9 * self._chunksize and self._step1:
            old = self._chunk_n
            new = self._chunk_n + 1
            self._step1=False
            self._start_chunk(old, new)
            
        if (self._frame_n % self._chunksize) == 0 or (self._frame_n == 1 and self._chunk_n == 0 and not self._already_init):
            old = self._chunk_n
            if self._chunk_n == 0 and not self._already_init:
                self._already_init = True
                self._frame_n = 0
                self._t0 = frame_time
                new = 0

            else:
                new = self._chunk_n + 1

            print(self._frame_n)

            #self._finish_chunk(old)
            #self._step1=True
            #self._capfn = self._capfn_
            #self._capfn_hq = self._capfn_hq_
            #self._cap = self._cap_
            #self._cap_hq = self._cap_hq_
            #self._new_chunk_metadata(os.path.join(self._basedir, '%06d' % new))
            #self.frame_idx = 0
            #self._chunk_n = new

            if not start_next_chunk:
                new = None
            new_fn = self._save_chunk(old, new)

        self.frame_idx += 1
        self.frame_count = self._frame_n
        return new_fn 

    def _save_miscoded_frame(self, img, chunk, frame_number):
        path = self._get_miscoded_frame_path(self._basedir, chunk, frame_number)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        cv2.imwrite(path, img)
        
    @staticmethod
    def _get_miscoded_frame_path(basedir, chunk, frame_number):
        return os.path.join(basedir, "miscoded", str(chunk).zfill(6), str(frame_number).zfill(10) + ".tiff")


    def _init_write(self, imgshape, imgdtype, chunksize, metadata, encoding, write_encode_encoding, fmt):
        for e in (encoding, write_encode_encoding):
            if e:
                if not self._codec_proc.check_code(e):
                    raise ValueError('unsupported store image encoding: %s' % e)

        # if encoding is unset, autoconvert is no-op
        print(f"write_encode_encoding: {write_encode_encoding}")
        self._codec_proc.set_default_code(write_encode_encoding)
        self._encode_image = self._codec_proc.autoconvert

        self._write_imgshape = self._calculate_image_shape(imgshape, fmt)

        if write_encode_encoding:
            # as we always encode to color
            self._write_imgshape = [self._write_imgshape[0], self._write_imgshape[1], 3]

        if not os.path.exists(self._basedir):
            os.makedirs(self._basedir)

        self._imgshape = imgshape
        self._imgdtype = imgdtype
        self._chunksize = chunksize
        self._format = fmt

        self._uuid = uuid.uuid4().hex
        # because fuck you python that utcnow is naieve. kind of fixed in python >3.2
        self._created_utc = datetime.datetime.utcnow().replace(tzinfo=pytz.utc)
        self._timezone_local = tzlocal.get_localzone()

        store_md = {'imgshape': self._write_imgshape,
                    'imgdtype': self._imgdtype,
                    'chunksize': chunksize,
                    'format': fmt,
                    'class': self.__class__.__name__,
                    'version': self._version,
                    'encoding': encoding,
                    # actually write the string as naieve because we have guarenteed it is UTC
                    'created_utc': self._created_utc.replace(tzinfo=None).isoformat(),
                    'timezone_local': str(self._timezone_local),
                    'uuid': self._uuid,
                    **self._metadata
                    }

        if metadata is None:
            metadata = {STORE_MD_KEY: store_md}
        elif isinstance(metadata, dict):
            try:
                metadata[STORE_MD_KEY].update(store_md)
            except KeyError:
                metadata[STORE_MD_KEY] = store_md
        else:
            raise ValueError('metadata must be a dictionary')

        with open(os.path.join(self._basedir, STORE_MD_FILENAME), 'wt') as f:
            yaml.safe_dump(metadata, f)

        with open(os.path.join(self._basedir, STORE_LOCK_FILENAME), 'a') as _:
            pass

        smd = metadata.pop(STORE_MD_KEY)
        self._metadata = smd
        self._user_metadata.update(metadata)

        self._save_chunk(None, self._chunk_n)


    def _save_chunk_metadata(self, path_without_extension, extension='.npz'):
        path = path_without_extension + extension
        self._save_index(path, self._chunk_md)

        # also calculate the filename of the extra file to hold any data
        if self._extra_data_fp is not None:
            self._extra_data_fp.write(']')
            self._extra_data_fp.close()
            self._extra_data_fp = None

    def _save_image_metadata(self, frame_number, frame_time):
        self._chunk_md['frame_number'].append(frame_number)
        self._chunk_md['frame_time'].append(frame_time)

    def save_index(self, path=None):
        if self._mode == 'r':
            if path is None:
                path = os.path.join(self._basedir, STORE_INDEX_FILENAME)
            self._index.to_file(path)
