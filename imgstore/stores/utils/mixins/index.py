import os.path
import json

import yaml
import pandas as pd
import numpy as np

class IndexMixin:

    @staticmethod
    def _save_index(path_with_extension, data_dict):
        _, extension = os.path.splitext(path_with_extension)
        if extension == '.yaml':
            with open(path_with_extension, 'wt') as f:
                yaml.safe_dump(data_dict, f)
                return path_with_extension
        elif extension == '.npz':
            with open(path_with_extension, 'wb') as f:
                # noinspection PyTypeChecker
                np.savez(f, **data_dict)
                return path_with_extension
        else:
            raise ValueError('unknown index format: %s' % extension)

    @staticmethod
    def _remove_index(path_without_extension):
        for extension in ('.npz', '.yaml'):
            path = path_without_extension + extension
            if os.path.exists(path):
                os.unlink(path)
                return path


    def reindex(self):
        """ modifies the current imgstore so that all framenumbers before frame_number=0 are negative

        if there are multiple frame_numbers equal to zero then this operation aborts. this functions also
        updates the frame_number of any stored extra data. the original frame_number prior to calling
        reindex() is stored in '_frame_number_before_reindex' """
        md = self.get_frame_metadata()
        fn = md['frame_number']

        nzeros = fn.count(0)
        if nzeros != 1:
            raise ValueError("%d frame_number=0 found (should be 1)" % nzeros)

        # get index and time of sync frame
        zero_idx = fn.index(0)
        self._log.info('reindexing about frame_number=0 at index=%d' % zero_idx)

        fn_new = fn[:]
        fn_new[:zero_idx] = range(-zero_idx, 0)

        for chunk_n, chunk_path in self._iter_chunk_n_and_chunk_paths():
            ind = self._index.get_chunk_metadata(chunk_n)

            oft = list(ind['frame_time'])

            ofn = list(ind['frame_number'])
            nfn = fn_new[chunk_n*self._chunksize:(chunk_n*self._chunksize) + self._chunksize]
            assert len(ofn) == len(nfn)

            new_ind = {'frame_time': oft,
                       'frame_number': nfn,
                       '_frame_number_before_reindex': ofn}

            _path = self._remove_index(chunk_path)
            self._log.debug('reindex chunk %s removed: %s' % (chunk_n, _path))
            _path = self._save_index(chunk_path + '.npz', new_ind)
            self._log.debug('reindex chunk %s wrote: %s' % (chunk_n, _path))

            for ext in ('.extra.json', '.extra_data.json'):
                ed_path = chunk_path + ext
                if os.path.exists(ed_path):
                    with open(ed_path, 'r') as f:
                        ed = json.load(f)

                    # noinspection PyBroadException
                    try:
                        df = pd.DataFrame(ed)
                        if 'frame_index' not in df.columns:
                            raise ValueError('can not reindex extra-data on old format stores')
                        df['_frame_number_before_reindex'] = df['frame_number']
                        df['frame_number'] = df.apply(lambda r: fn_new[int(r.frame_index)], axis=1)
                        with open(ed_path, 'w') as f:
                            df.to_json(f, orient='records')
                        self._log.debug('reindexed chunk %d metadata (%s)' % (chunk_n, ed_path))
                    except Exception:
                        self._log.error('could not update chunk extra data to new framenumber', exc_info=True)