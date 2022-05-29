import os.path
import yaml
from imgstore.constants import STORE_MD_KEY

def _extract_store_metadata(full_path):
    with open(full_path, 'r') as f:
        allmd = yaml.load(f, Loader=yaml.SafeLoader)
    return allmd.pop(STORE_MD_KEY)



class ExtractMixin:

    @staticmethod
    def _extract_only_frame(basedir, chunk_n, frame_n, smd):
        return None

    @classmethod
    def extract_only_frame(cls, full_path, frame_index, _smd=None):
        if _smd is None:
            smd = _extract_store_metadata(full_path)
        else:
            smd = _smd

        chunksize = int(smd['chunksize'])

        # go directly to the chunk
        chunk_n = frame_index // chunksize
        frame_n = frame_index % chunksize

        return cls._extract_only_frame(basedir=os.path.dirname(full_path),
                                       chunk_n=chunk_n,
                                       frame_n=frame_n,
                                       smd=smd)