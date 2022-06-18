from __future__ import print_function
import tempfile
import numpy as np
import numpy.testing as npt
import os.path
import shutil
import tempfile
import time
import itertools
import pickle
import warnings
import traceback

import cv2
import pytest

from imgstore import stores
from imgstore.util import FourCC, ensure_color, ensure_grayscale
from imgstore.tests import TEST_DATA_DIR


def encode_image(num, nbits=16, imgsize=512):
    # makes a square image that looks a bit like a barcode. colums of the matrix all 255
    # if the bit in the number is 1, 0 otherwise
    row = np.fromiter((255*int(c) for c in ('{0:0%db}' % nbits).format(num)), dtype=np.uint8)
    mat = np.tile(row, (nbits, 1))
    return cv2.resize(mat, dsize=(imgsize, imgsize), interpolation=cv2.INTER_NEAREST)

@pytest.mark.parametrize('fmt', stores.get_supported_formats())
def test_serialize(tmpdir, fmt):

    nframes=10
    assert os.path.isdir(tmpdir.strpath)
    
    # assert False

    with stores.new_for_format(
        fmt=fmt,
        basedir=tmpdir.strpath,
        imgshape=(512, 512),
        imgdtype=np.uint8,
        chunksize=13
    ) as s:
        for fn in range(nframes):
            s.add_image(encode_image(fn, imgsize=512), fn, time.time())

    # import warnings
    # if "_codec" in s.__dict__: warnings.warn(f"Codec: {s._codec}")

    d = stores.new_for_filename(s.full_path)

    # test they can be serialized   
    s.save(os.path.join(tmpdir.strpath, "writing_mode.pkl"))
    d.save(os.path.join(tmpdir.strpath, "reading_mode.pkl"))

    try:
        ss=s.__class__.load(os.path.join(tmpdir.strpath, "writing_mode.pkl"))
    except Exception as error:
        warnings.warn(error, stacklevel=2)
        warnings.warn(traceback.print_exc(), stacklevel=2)
    dd=d.__class__.load(os.path.join(tmpdir.strpath, "reading_mode.pkl"))
    
    img_d, _ = d.get_image(5)
    img_dd, _ = dd.get_image(5)
    assert np.all(img_d == img_dd)
    print("run")

