import abc
import os.path
import datetime
import pytz
import tzlocal
import uuid

import yaml
import numpy as np

from imgstore.constants import STORE_MD_FILENAME, STORE_LOCK_FILENAME, STORE_MD_KEY, STORE_INDEX_FILENAME

class WritingStore(abc.ABC):


    def add_image(self, img, frame_number, frame_time):

        if img.shape != self._write_imgshape:
            img = img[:self._write_imgshape[0], :self._write_imgshape[1]].copy(order="C")
        self._save_image(self._encode_image(img), frame_number, frame_time)

        self.frame_max = np.nanmax((frame_number, self.frame_max))
        self.frame_min = np.nanmin((frame_number, self.frame_min))
        self.frame_number = frame_number
        self.frame_time = frame_time

        if self._frame_n == 0:
            self._t0 = frame_time
        self._tN = frame_time

        self._frame_n += 1
        if (self._frame_n % self._chunksize) == 0:
            old = self._chunk_n
            new = self._chunk_n + 1
            self._save_chunk(old, new)
            self._chunk_n = new

        self.frame_count = self._frame_n


    def _init_write(self, imgshape, imgdtype, chunksize, metadata, encoding, write_encode_encoding, fmt):
        for e in (encoding, write_encode_encoding):
            if e:
                if not self._codec_proc.check_code(e):
                    raise ValueError('unsupported store image encoding: %s' % e)

        # if encoding is unset, autoconvert is no-op
        self._codec_proc.set_default_code(write_encode_encoding)
        self._encode_image = self._codec_proc.autoconvert

        self._write_imgshape = self._calculate_image_shape(imgshape, fmt)
        # self._write_imgshape=imgshape


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
