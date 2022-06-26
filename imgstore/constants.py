import os.path

try:
    # python 3
    # noinspection PyProtectedMember
    from subprocess import DEVNULL
    # noinspection PyShadowingBuiltins
    xrange = range
except ImportError:
    # python 2
    import os
    DEVNULL = open(os.devnull, 'r+b')

STORE_MD_KEY = '__store'
STORE_MD_FILENAME = 'metadata.yaml'
STORE_LOCK_FILENAME = '.lock'
STORE_INDEX_FILENAME = '.index.sqlite'

EXTRA_DATA_FILE_EXTENSIONS = ('.extra.json', '.extra_data.json', '.extra_data.h5')

FRAME_MD = ('frame_number', 'frame_time')
VERBOSE_DEBUG_CHUNKS=False
CONFIG_FILE=os.path.join(os.environ["HOME"], ".config", "imgstore", "imgstore.yml")
SQLITE3_INDEX_FILE="index.db"
FRAME_NUMBER_RESET=True
MULTI_STORE_ENABLED=False
COLOR=True
SELECTED_STORE="lowres/metadata.yaml"
LOOKUP_NEAREST=False