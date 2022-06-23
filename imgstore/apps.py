from __future__ import print_function, absolute_import
import logging

import cv2
import numpy as np

from .stores import new_for_filename, get_supported_formats, new_for_format
from .ui import new_window
from .util import motif_get_parse_true_fps

from imgstore.stores.utils.formats import get_formats


_log = logging.getLogger('imgstore.apps')


def main_viewer():
    import argparse
    import logging

    logging.basicConfig(level=logging.DEBUG)

    parser = argparse.ArgumentParser()
    parser.add_argument('path', nargs=1)
    parser.add_argument('--fps', default=30., type=float,
                        help='playback store at this speed (0 = play back as fast as possible)')
    args = parser.parse_args()

    if args.fps == 0.0:
        args.fps = 1000.0  # ensure w sleep at least 1ms

    store = new_for_filename(args.path[0])

    _log.info('true fps: %s' % motif_get_parse_true_fps(store))
    _log.info('fr fps: %s' % (1.0 / np.median(np.diff(store._get_chunk_metadata(0)['frame_time']))))

    win = new_window('imgstore', shape=store.image_shape)
    _log.debug('created window: %r' % win)

    while True:
        try:
            img, _ = store.get_next_image()
        except EOFError:
            break

        win.imshow('imgstore', img)

        k = cv2.waitKey(int(1000. / args.fps)) & 0xFF
        if k == 27:
            break


def main_saver():
    import os.path
    import argparse
    import logging
    import time
    import itertools
    import errno

    logging.basicConfig(level=logging.DEBUG)

    parser = argparse.ArgumentParser()
    parser.add_argument('dest', nargs=1)
    parser.add_argument('--source', type=str, default=0,
                        metavar='PATH',
                        help='path to a file to convert to an imgstore')
    parser.add_argument('--format',
                        choices=get_supported_formats(), default='mjpeg')
    args = parser.parse_args()

    # noinspection PyArgumentList
    cap = cv2.VideoCapture(args.source)
    _, img = cap.read()

    if img is None:
        parser.error('could not open source: %s' % args.source)

    path = args.dest[0]
    if os.path.exists(path):
        parser.error('destination already exists')
    else:
        try:
            os.makedirs(path)
        except OSError as exc:  # Python >2.5
            if exc.errno == errno.EEXIST and os.path.isdir(path):
                pass
            else:
                parser.error('could not create destination path')

    store = new_for_format(args.format,
                           basedir=path, mode='w',
                           imgdtype=img.dtype,
                           imgshape=img.shape)

    for i in itertools.count():
        _, img = cap.read()

        cv2.imshow('preview', img)

        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

        store.add_image(img, i, time.time())

    store.close()


def main_test():
    import pytest
    import os.path

    pytest.main(['-v', os.path.join(os.path.abspath(os.path.dirname(__file__)), 'tests')])

def list_codecs():
    formats=get_formats(video=True, directory=True, raw=True, cache=False)
    formats="\n".join([f for f in formats if formats[f]])
    
    return formats

def generate_timecodes(store, dest_file):
    _ts = np.asarray(store.get_frame_metadata()['frame_time'])
    ts = _ts - _ts[0]

    dest_file.write('# timecode format v2\n')
    for t in ts:
        dest_file.write('{0:.3f}\n'.format(t * 1000.))
    return ts


def main_generate_timecodes():
    import sys
    import argparse
    import logging

    logging.basicConfig(level=logging.DEBUG)

    parser = argparse.ArgumentParser()
    parser.add_argument('path', nargs=1)
    args = parser.parse_args()

    store = new_for_filename(args.path[0])

    generate_timecodes(store, sys.stdout)


def main_muxer():

    import argparse
    import os.path
    import cv2
    from imgstore.stores import new_for_format

    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str)
    parser.add_argument("-s", type=str, default=None, help="""
        XX:XX:XX position from which to start the video
    """
    )
    parser.add_argument("-t", type=str, default=None, help="""
        XX:XX:XX duration of video to be muxed
    """
    )

    parser.add_argument("--output", type=str, help="""
    Path to a metadata.yaml file in a folder
    which will serve as the directory
    where the imgstore will be created
    """)

    args = parser.parse_args()

    assert os.path.exists(args.video)
    assert not os.path.exists(args.output)
    assert os.path.basename(args.output) == "metadata.yaml"

    cap = cv2.VideoCapture(args.video)

    store = new_for_format(
        fmt="h264_nvenc/mp4", path=args.output,
        chunksize=100, fps=25    
    )

    if args.s:
        hour, minute, second = args.s.split(":")
        msecs = hour*3600*1000 + minute*60*1000 + second*1000
        cap.set(0, msecs)

    if args.t:
        hour, minute, second = args.t.split(":")
        duration = hour*3600*1000 + minute*60*1000 + second*1000

    ft0 = cap.get(0)
    ft=ft0
    fn = cap.get(1)

    ret, img = cap.read()
    while ret and (cap.get(0) - ft0) < duration:
        store.add_image(img, frame_number=fn, frame_time=ft)
        ft = cap.get(0)
        fn = cap.get(1)
        ret, img = cap.read()

    cap.release()
    store.release()