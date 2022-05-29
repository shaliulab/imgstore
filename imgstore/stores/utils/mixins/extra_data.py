import os.path
import json
from imgstore.util import JsonCustomEncoder
from imgstore.constants import EXTRA_DATA_FILE_EXTENSIONS

class ExtraDataMixin:

    @property
    def has_extra_data(self):
        for _ in self._iter_extra_data_files():
            return True
        return False


    def add_extra_data(self, **data):
        if not data:
            return

        data['frame_time'] = self.frame_time
        data['frame_number'] = self.frame_number
        data['frame_index'] = self._frame_n - 1  # we post-increment in add_frame

        # noinspection PyBroadException
        try:
            txt = json.dumps(data, cls=JsonCustomEncoder)
        except Exception:
            self._log.warn('error writing extra data', exc_info=True)
            return

        if self._extra_data_fp is None:
            self._extra_data_fp = open(self._extra_data_fn, 'w')
            self._extra_data_fp.write('[')
        else:
            self._extra_data_fp.write(', ')
        self._extra_data_fp.write(txt)

    def _new_chunk_metadata(self, chunk_path):
        self._extra_data_fn = chunk_path + '.extra.json'
        self._chunk_md = {k: [] for k in self.FRAME_MD}
        self._chunk_md.update(self._metadata)


    def _iter_extra_data_files(self, extensions=EXTRA_DATA_FILE_EXTENSIONS):
        for chunk_n, chunk_path in self._iter_chunk_n_and_chunk_paths():
            out = None

            # motif out of process IO puts extra data in the root directory
            for ext in ('.extra_data.h5', '.extra_data.json'):

                if ext not in extensions:
                    continue

                path = os.path.join(self._basedir, '%06d%s' % (int(chunk_n), ext))
                if os.path.exists(path):
                    yield True, path
                    break

            else:
                # for .. else
                #
                # if we _didn't_ break above then we enter here

                # imgstore API puts them beside the index,
                # which is a subdir for dirimgstores
                for ext in ('.extra.json', '.extra_data.json'):

                    if ext not in extensions:
                        continue

                    path = chunk_path + ext
                    if os.path.exists(path):
                        yield False, path
