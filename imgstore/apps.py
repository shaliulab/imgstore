from __future__ import print_function, absolute_import
import logging
import os.path
import cv2
import numpy as np
import shutil

from .stores import (
    new_for_filename,
    get_supported_formats,
    read_metadata,
    new_for_format
)

from .ui import new_window
from .util import motif_get_parse_true_fps
from .multistores import MultiStore
import cv2cuda


_log = logging.getLogger("imgstore.apps")


def main_viewer():
    import argparse
    import logging

    logging.basicConfig(level=logging.DEBUG)

    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs=1)
    parser.add_argument(
        "--fps",
        default=30.0,
        type=float,
        help="playback store at this speed (0 = play back as fast as possible)",
    )
    args = parser.parse_args()

    if args.fps == 0.0:
        args.fps = 1000.0  # ensure w sleep at least 1ms

    store = new_for_filename(args.path[0])

    _log.info("true fps: %s" % motif_get_parse_true_fps(store))
    _log.info(
        "fr fps: %s"
        % (
            1.0
            / np.median(np.diff(store._get_chunk_metadata(0)["frame_time"]))
        )
    )

    win = new_window("imgstore", shape=store.image_shape)
    _log.debug("created window: %r" % win)

    while True:
        try:
            img, _ = store.get_next_image()
        except EOFError:
            break

        win.imshow("imgstore", img)

        k = cv2.waitKey(int(1000.0 / args.fps)) & 0xFF
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
    parser.add_argument("dest", nargs=1)
    parser.add_argument(
        "--source",
        type=str,
        default=0,
        metavar="PATH",
        help="path to a file to convert to an imgstore",
    )
    parser.add_argument(
        "--format", choices=get_supported_formats(), default="mjpeg"
    )
    args = parser.parse_args()

    # noinspection PyArgumentList
    cap = cv2.VideoCapture(args.source)
    _, img = cap.read()

    if img is None:
        parser.error("could not open source: %s" % args.source)

    path = args.dest[0]
    if os.path.exists(path):
        parser.error("destination already exists")
    else:
        try:
            os.makedirs(path)
        except OSError as exc:  # Python >2.5
            if exc.errno == errno.EEXIST and os.path.isdir(path):
                pass
            else:
                parser.error("could not create destination path")

    store = new_for_format(
        args.format,
        basedir=path,
        mode="w",
        imgdtype=img.dtype,
        imgshape=img.shape,
    )

    for i in itertools.count():
        _, img = cap.read()

        cv2.imshow("preview", img)

        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

        store.add_image(img, i, time.time())

    store.close()


def main_test():
    import pytest
    import os.path

    pytest.main(
        [
            "-v",
            os.path.join(os.path.abspath(os.path.dirname(__file__)), "tests"),
        ]
    )


def generate_timecodes(store, dest_file):
    _ts = np.asarray(store.get_frame_metadata()["frame_time"])
    ts = _ts - _ts[0]

    dest_file.write("# timecode format v2\n")
    for t in ts:
        dest_file.write("{0:.3f}\n".format(t * 1000.0))
    return ts


def main_generate_timecodes():
    import sys
    import argparse
    import logging

    logging.basicConfig(level=logging.DEBUG)

    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs=1)
    args = parser.parse_args()

    store = new_for_filename(args.path[0])

    generate_timecodes(store, sys.stdout)


def multistore_index_parser(ap=None):
    import argparse

    if ap is None:
        ap = argparse.ArgumentParser()
    
    ap.add_argument("--input", help="Path to imgstore folder or metadata.yaml")
    ap.add_argument("--ref-chunk", dest="ref_chunk", default=0, type=int)
    return ap


def main_multistore_index(ap=None, args=None):

    if args is None:
        ap = multistore_index_parser(ap)
        args = ap.parse_args()
    
    store = MultiStore.new_for_filename(
        args.input,
        ref_chunk=args.ref_chunk
    )

    store.export_index_to_csv()

def clip_video(input, chunk, interval, extension):
    INPUT_VIDEO = os.path.join(input, str(chunk).zfill(6) + extension)
    OUTPUT_VIDEO = os.path.join(input, "clips", str(chunk).zfill(6) + extension)

    assert os.path.exists(INPUT_VIDEO)

    cap = cv2.VideoCapture(
        INPUT_VIDEO
    )

    cap.set(cv2.CAP_PROP_POS_FRAMES, interval[0])
    frameSize=(
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    )

    video_writer = cv2cuda.VideoWriter(
        filename=OUTPUT_VIDEO,
        apiPreference="FFMPEG",
        fourcc="h264_nvenc",
        fps=cap.get(cv2.CAP_PROP_FPS),
        frameSize=frameSize,
        isColor=False,
    )

    while True:

        ret, frame = cap.read()

        if len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        video_writer.write(frame)

        if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) == interval[1]:
            break

    
    cap.release()
    video_writer.release()

def clip_index(input, chunk, interval, extension=".npz"):

    INPUT_INDEX = os.path.join(input, str(chunk).zfill(6) + extension)
    OUTPUT_INDEX = os.path.join(input, "clips", str(chunk).zfill(6) + extension)

    data = np.load(INPUT_INDEX)
    data.allow_pickle=True
    data=dict(data)

    data_dict = data.copy()
    data_dict = {
        "frame_in_chunk": data["frame_in_chunk"][interval[0]:interval[1]],
        "frame_time": data["frame_time"][interval[0]:interval[1]],
        "frame_number": data["frame_number"][interval[0]:interval[1]],
    }

    with open(OUTPUT_INDEX, "wb") as f:
        # noinspection PyTypeChecker
        np.savez(f, **data_dict)


def clip_chunk(ap=None, args=None):

    if args is None:
        ap = multistore_index_parser(ap)
        ap.add_argument("--interval", nargs=2, type=int)
        args = ap.parse_args()

    _, metadata = read_metadata(args.input)
    extension = metadata["extension"]

    os.makedirs(
        os.path.join(
            args.input, "clips"
        ),
        exist_ok=True
    )

    _log.info("Clipping video...")
    clip_video(input=args.input, chunk=args.ref_chunk, interval=args.interval, extension=extension)
    _log.info("Clipping index...")
    clip_index(input=args.input, chunk=args.ref_chunk, interval=args.interval, extension=".npz")

    _log.info("Copying json and png")
    shutil.copyfile(
        os.path.join(
            args.input, str(args.ref_chunk).zfill(6) + ".extra.json"
        ),
        os.path.join(
            args.input, "clips", str(args.ref_chunk).zfill(6) + ".extra.json"
        )
    )

    shutil.copyfile(
        os.path.join(
            args.input, str(args.ref_chunk).zfill(6) + ".png"
        ),
        os.path.join(
            args.input, "clips", str(args.ref_chunk).zfill(6) + ".png"
        )
    )
