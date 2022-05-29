import os.path
from imgstore.constants import STORE_MD_FILENAME

def get_fullpath(basedir):
    return os.path(basedir, STORE_MD_FILENAME)