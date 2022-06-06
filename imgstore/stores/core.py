import os.path
import yaml
from imgstore.constants import STORE_MD_FILENAME, STORE_MD_KEY
from imgstore.stores.video import VideoImgStore
from imgstore.stores.directory import DirectoryImgStore
from imgstore.stores.utils.mixins.extract import _extract_store_metadata

def _parse_basedir_fullpath(path, basedir, read):
    # the API is not so clean that we support both path and basedir, but we have to keep
    # backwards compatibility as best as possible

    if path and basedir:
        raise ValueError('path and basedir are mutually exclusive')

    if basedir:
        if basedir.endswith(STORE_MD_FILENAME) and (os.path.basename(basedir) == STORE_MD_FILENAME):
            raise ValueError('basedir should be a directory (not a file ending with %s)' % STORE_MD_FILENAME)
        return basedir, os.path.join(basedir, STORE_MD_FILENAME)

    if path and path.endswith(STORE_MD_FILENAME) and (os.path.basename(path) == STORE_MD_FILENAME):
        basedir = os.path.dirname(path)
        fullpath = path
    elif path:
        if read:
            if os.path.isdir(path) and os.path.exists(os.path.join(path, STORE_MD_FILENAME)):
                basedir = path
                fullpath = os.path.join(path, STORE_MD_FILENAME)
            else:
                raise ValueError("path '%s' does not exist" % path)
        else:
            # does not end with STORE_MD_FILENAME
            basedir = path
            fullpath = os.path.join(path, STORE_MD_FILENAME)
    else:
        raise ValueError('should be a path to a store %s file or a directory containing one' % STORE_MD_FILENAME)

    return basedir, fullpath


def new_for_filename(path, **kwargs):
    if 'mode' not in kwargs:
        kwargs['mode'] = 'r'

    basedir, fullpath = _parse_basedir_fullpath(path, kwargs.pop('basedir', None),
                                                read=kwargs.get('mode') == 'r')

    kwargs['basedir'] = basedir

    with open(fullpath, 'rt') as f:
        clsname = yaml.load(f, Loader=yaml.SafeLoader)[STORE_MD_KEY]['class']

    # retain compatibility with internal loopbio stores
    if clsname == 'VideoImgStoreFFMPEG':
        clsname = 'VideoImgStore'

    try:
        cls = {DirectoryImgStore.__name__: DirectoryImgStore,
               VideoImgStore.__name__: VideoImgStore}[clsname]
    except KeyError:
        raise ValueError('store class %s not supported' % clsname)

    return cls(**kwargs)


def new_for_format(fmt, path=None, **kwargs):
    if 'mode' not in kwargs:
        kwargs['mode'] = 'w'

    if kwargs.get('mode') == 'r':
        return new_for_filename(path, **kwargs)

    # we are writing mode
    basedir, _ = _parse_basedir_fullpath(path, kwargs.pop('basedir', None),
                                         read=kwargs.get('mode') == 'r')
    kwargs['basedir'] = basedir

    for cls in (DirectoryImgStore, VideoImgStore):
        if cls.supports_format(fmt):
            kwargs['format'] = fmt
            return cls(**kwargs)
    raise ValueError("store class not found which supports format '%s'" % fmt)


def extract_only_frame(path, frame_index):

    _, fullpath = _parse_basedir_fullpath(path, None, True)

    smd = _extract_store_metadata(fullpath)
    clsname = smd['class']

    if clsname == 'VideoImgStoreFFMPEG':
        clsname = 'VideoImgStore'

    try:
        cls = {DirectoryImgStore.__name__: DirectoryImgStore,
               VideoImgStore.__name__: VideoImgStore}[clsname]
    except KeyError:
        raise ValueError('store class %s not supported' % clsname)

    return cls.extract_only_frame(full_path=fullpath, frame_index=frame_index, _smd=smd)


def get_supported_formats():
    f = []
    for cls in (DirectoryImgStore, VideoImgStore):
        f.extend(cls.supported_formats())
    return f
