import abc
import yaml
import json
import dateutil.parser

import pandas as pd

from imgstore.constants import STORE_MD_KEY, EXTRA_DATA_FILE_EXTENSIONS
from imgstore.util import motif_extra_data_h5_to_df, motif_extra_data_json_to_df, motif_extra_data_h5_attrs
from imgstore.stores.utils.datetime import parse_old_time, parse_new_time
from imgstore.stores.utils.path import get_fullpath

class ReadingStore(abc.ABC):

    def _init_read(self):
        """
        
        """
        
        fullpath = get_fullpath(self._basedir)
        with open(fullpath, 'r') as f:
            allmd = yaml.load(f, Loader=yaml.SafeLoader)
        smd = allmd.pop(STORE_MD_KEY)

        self._metadata = smd
        self._user_metadata.update(allmd)

        self._readability_check(smd_class=smd['class'],
                                smd_version=smd['version'])

        # noinspection PyShadowingNames
        uuid = smd.get('uuid', None)
        self._uuid = uuid

        self._imgshape = tuple(smd['imgshape'])
        self._imgdtype = smd['imgdtype']
        self._chunksize = int(smd['chunksize'])
        self._encoding = smd['encoding']
        self._format = smd['format']

        # synthesize a created_date from old format stores
        if 'created_utc' not in smd:
            self._log.info('old store detected. synthesizing created datetime / timezone')
            dt, tz = parse_old_time(allmd, self._basedir)
        else:
            _dt = dateutil.parser.parse(smd['created_utc'])
            self._log.debug('parsed created_utc: %s (from %r)' % (_dt.isoformat(), _dt))
            dt, tz = parse_new_time(smd, _dt)

        self._created_utc = dt
        self._timezone_local = tz

        # if encoding is unset, autoconvert is no-op
        self._codec_proc.set_default_code(self._encoding)
        self._decode_image = self._codec_proc.autoconvert


    def _readability_check(self, smd_class, smd_version):
        """
        Raise Value error if name of this Python class or its version
        does not match the passed name and version,
        and True otherwise i.e. if they match
        """
        class_name = getattr(self, 'class_name', self.__class__.__name__)
        if smd_class != class_name:
            raise ValueError('incompatible store, can_read:%r opened:%r' % (class_name, smd_class))
        if smd_version != self._version:
            raise ValueError('incompatible store version')
        return True


    def find_extra_data_files(self, extensions=EXTRA_DATA_FILE_EXTENSIONS):
        return [path for _, path in self._iter_extra_data_files(extensions=extensions)]

    def get_extra_data_samplerate(self):
        for c in self.find_extra_data_files(extensions=('.extra_data.h5',)):
            attrs = motif_extra_data_h5_attrs(c)
            ds = attrs['_datasets']
            if len(ds) > 1:
                raise ValueError('Multiple sample rates detected: please use motif_extra_data_h5_attrs')
            elif len(ds) == 1:
                try:
                    return attrs[ds[0]]['samplerate']
                except KeyError:
                    pass

        raise ValueError("No sample rate or dataset found in file")

    def get_extra_data(self, ignore_corrupt_chunks=False):
        dfs = []

        for is_motif, path in self._iter_extra_data_files():

            self._log.debug('found extra data chunk: %s (motif: %s)' % (path, is_motif))

            # noinspection PyBroadException
            try:
                if is_motif:
                    if path.endswith('.h5'):
                        df = motif_extra_data_h5_to_df(self, path)
                    elif path.endswith('.json'):
                        df = motif_extra_data_json_to_df(self, path)
                    else:
                        df = None

                    if (df is not None) and (not df.empty):
                        dfs.append(df)
                    else:
                        self._log.warn('chunk extra data %s is empty' % path)

                else:
                    if path.endswith('.json'):
                        with open(path, 'rt') as f:
                            records = json.load(f)
                        dfs.append(pd.DataFrame(records))
                    else:
                        raise ValueError('unknown extra data file encountered')

            except Exception:
                if ignore_corrupt_chunks:
                    self._log.warn('chunk extra data %s is corrupt' % path, exc_info=True)
                    continue
                else:
                    raise
        if dfs:
            return pd.concat(dfs, axis=0, ignore_index=True)
