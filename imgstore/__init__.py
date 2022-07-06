from .stores import new_for_filename, new_for_format, extract_only_frame, get_supported_formats,\
    VideoImgStore, DirectoryImgStore
from .apps import main_test
from .util import ensure_color, ensure_grayscale
from .constants import LOGGING_FILE

test = main_test
from confapp import conf

conf += "imgstore.constants"

import logging
import logging.config
import yaml
import os.path
if os.path.exists(LOGGING_FILE):
    with open(LOGGING_FILE, "r") as filehandle:
        config=yaml.load(filehandle, yaml.SafeLoader)

    logging.config.dictConfig(config)

__version__ = '0.4.6'
