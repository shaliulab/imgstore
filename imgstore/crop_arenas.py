"""
Split an imgstore recording into one imgstore recording per arena (ROI)

    imgstore-crop-arenas --input path/to/metadata.yaml --arenas path/to/arenas.toml

produces path/to/arenas/1/metadata.yaml, path/to/arenas/2/metadata.yaml, ...
Each new recording holds the bounding box of one ROI and has the same chunks
(and .npz frame number / frame time indices) as the original recording.

ROIs are read from the idtrackerai roi_list of the arenas.toml file

    roi_list = [
        "+ Polygon [[x, y], [x, y], ...]",
        "+ Rectangle [[x0, y0], [x1, y1]]",
        "+ Ellipse {'center': [x, y], 'axes': [a, b], 'angle': 0}",
    ]

Alternatively, one table per arena with either a polygon or x, y, width, height

    [[arenas]]
    x = 0
    y = 0
    width = 500
    height = 500

    [[arenas]]
    polygon = [[x, y], [x, y], ...]

Pass --size 1024 to give all arena videos the same size: each ROI is cropped to its
bounding box and padded with black to 1024x1024, with the ROI center in the center
of the frame (the pixel scale is not changed). The arena block of each metadata.yaml
has the offset to map arena coordinates back to the original frame:

    original (x, y) = arena (x, y) + offset

Chunks are re-encoded with the same codec as the source chunks (crf 12 by default, or --lossless)

Requires ffmpeg and ffprobe in the PATH
"""

import argparse
import ast
import concurrent.futures
import glob
import json
import logging
import os.path
import re
import shutil
import subprocess
import uuid

import cv2
import numpy as np
import tqdm
import yaml

try:
    import tomllib

    def _load_toml(path):
        with open(path, "rb") as filehandle:
            return tomllib.load(filehandle)
except ImportError:
    import toml

    def _load_toml(path):
        return toml.load(path)

from imgstore.constants import STORE_MD_KEY, STORE_MD_FILENAME
from imgstore.stores.utils.path import get_fullpath

logger = logging.getLogger(__name__)

POLYGON_KEYS = ("polygon", "points", "vertices", "contour", "roi")


def _parse_roi_string(roi):
    """
    Parse an idtrackerai ROI string such as "+ Polygon [[x, y], ...]",
    "+ Rectangle [[x0, y0], [x1, y1]]" or "+ Ellipse {'center': ..., 'axes': ..., 'angle': ...}"
    """
    roi = roi.strip()
    if roi.startswith("-"):
        raise ValueError(f"Negative (excluding) ROIs are not supported: {roi}")

    if "ellipse" in roi.lower():
        ellipse = ast.literal_eval(roi[roi.index("{"):])
        center = tuple(int(round(v)) for v in ellipse["center"])
        axes = tuple(int(round(v)) for v in ellipse["axes"])
        return cv2.ellipse2Poly(center, axes, int(round(ellipse.get("angle", 0))), 0, 360, 5).astype(np.int64)

    match = re.search(r"\[.*\]", roi, re.DOTALL)
    if match is None:
        raise ValueError(f"Cannot parse ROI {roi}")

    points = np.array(json.loads(match.group(0)), dtype=np.int64).reshape(-1, 2)
    if "rectangle" in roi.lower() and len(points) == 2:
        (x0, y0), (x1, y1) = points
        points = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
    return points


def _parse_roi(roi):
    if isinstance(roi, str):
        return _parse_roi_string(roi)

    if isinstance(roi, (list, tuple)):
        return np.array(roi, dtype=np.int64).reshape(-1, 2)

    for key in POLYGON_KEYS:
        if key in roi:
            return _parse_roi(roi[key])

    x, y = roi["x"], roi["y"]
    w = roi.get("width", roi.get("w"))
    h = roi.get("height", roi.get("h"))
    if w is None or h is None:
        raise ValueError(f"ROI {roi} needs x, y, width, height or a polygon")
    return np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.int64)


def load_arenas(path):
    """
    Return a list of polygons (Nx2 arrays of x, y coordinates), one per arena
    """
    config = _load_toml(path)

    if "roi_list" in config:
        rois = config["roi_list"]
    elif "arenas" in config:
        rois = config["arenas"]
    else:
        # one table per arena, keyed by its name
        tables = {k: v for k, v in config.items() if isinstance(v, dict)}
        rois = [tables[k] for k in sorted(tables, key=lambda k: (not k.isdigit(), int(k) if k.isdigit() else 0, k))]

    if not rois:
        raise ValueError(f"No ROIs found in {path}")

    return [_parse_roi(roi) for roi in rois]


def get_bbox(polygon, imgshape):
    """
    Bounding box of the polygon clipped to the frame, with even origin, width and height
    (required by yuv420p / h264)
    """
    height, width = imgshape[:2]
    x0, y0 = polygon.min(axis=0)
    x1, y1 = polygon.max(axis=0)
    x0, y0 = max(int(x0), 0), max(int(y0), 0)
    # ffmpeg crop rounds odd offsets down to even for chroma subsampled (yuv420p) video
    x0, y0 = x0 - x0 % 2, y0 - y0 % 2
    x1, y1 = min(int(x1), width), min(int(y1), height)
    # even width and height, growing the box rather than trimming the ROI when possible
    if (x1 - x0) % 2:
        x1 = x1 + 1 if x1 < width else x1 - 1
    if (y1 - y0) % 2:
        y1 = y1 + 1 if y1 < height else y1 - 1
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"ROI {polygon.tolist()} does not overlap with the frame ({width}x{height})")
    return x0, y0, x1, y1


def get_padding(polygon, bbox, size):
    """
    Return the left and top padding that places the center of the polygon
    at the center of a frame of the given (width, height),
    after cropping its bbox
    """
    x0, y0, x1, y1 = bbox
    width, height = size
    if (x1 - x0) > width or (y1 - y0) > height:
        raise ValueError(
            f"ROI {polygon.tolist()} ({x1 - x0}x{y1 - y0}) does not fit in {width}x{height}. Pass a larger --size"
        )

    center_x, center_y = (polygon.min(axis=0) + polygon.max(axis=0)) / 2
    pad_x = int(np.clip(round(width / 2 - (center_x - x0)), 0, width - (x1 - x0)))
    pad_y = int(np.clip(round(height / 2 - (center_y - y0)), 0, height - (y1 - y0)))
    # even offsets so the chroma planes (yuv420p) stay aligned
    return pad_x - pad_x % 2, pad_y - pad_y % 2


def _find_chunks(basedir, extension):
    """
    Return {chunk: video path}. Chunk 0 of h264_nvenc stores may only exist as .avi
    """
    chunks = {}
    for index in sorted(glob.glob(os.path.join(basedir, "[0-9]" * 6 + ".npz"))):
        chunk = int(os.path.basename(index)[:6])
        for ext in (extension, ".avi"):
            video = os.path.join(basedir, "%06d%s" % (chunk, ext))
            if os.path.exists(video):
                chunks[chunk] = video
                break
        else:
            logger.warning("Chunk %s has an index but no video. Skipping", chunk)
    return chunks


# CPU encoder used to re-encode each source codec (as reported by ffprobe)
ENCODERS = {
    "h264": "libx264",
    "hevc": "libx265",
    "mpeg4": "mpeg4",
    "mjpeg": "mjpeg",
}

# NVIDIA GPU encoder used instead with --gpu
GPU_ENCODERS = {
    "h264": "h264_nvenc",
    "hevc": "hevc_nvenc",
}


def probe_video(video):
    """
    Return codec_name, pix_fmt and profile of the first video stream
    """
    process = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=codec_name,pix_fmt,profile", "-of", "json", video],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    streams = json.loads(process.stdout or "{}").get("streams", [])
    if process.returncode != 0 or not streams:
        raise RuntimeError(f"Cannot probe {video}:\n{process.stderr}")
    return streams[0]


def get_encoder_args(stream, codec=None, crf=12, lossless=False, gpu=False):
    """
    ffmpeg output arguments to re-encode a stream like the source one,
    either losslessly or at high quality
    """
    codec_name = stream["codec_name"]
    if codec is not None:
        encoder = codec
    elif gpu and codec_name in GPU_ENCODERS:
        encoder = GPU_ENCODERS[codec_name]
    else:
        encoder = ENCODERS.get(codec_name, codec_name)

    if lossless and encoder in ("mpeg4", "mjpeg"):
        # these codecs cannot be lossless (e.g. chunk 0 of h264_nvenc stores is mpeg4)
        encoder = "h264_nvenc" if gpu else "libx264"
    args = ["-c:v", encoder]
    if stream.get("pix_fmt"):
        args += ["-pix_fmt", stream["pix_fmt"]]

    if encoder in ("libx264", "libx265"):
        if lossless:
            args += ["-qp", "0"] if encoder == "libx264" else ["-x265-params", "lossless=1"]
        else:
            args += ["-crf", str(crf), "-preset", "slow"]
        if encoder == "libx264" and not lossless and stream.get("profile", "").lower() in ("baseline", "main", "high"):
            args += ["-profile:v", stream["profile"].lower()]
    elif encoder.endswith("_nvenc"):
        if lossless:
            args += ["-tune", "lossless"]
        else:
            # constant QP: -rc vbr -cq caps the quality (~37 dB PSNR whatever the cq)
            # qp = crf - 2 gives the same PSNR as libx264 -crf -preset slow
            args += ["-rc", "constqp", "-qp", str(max(crf - 2, 0)), "-preset", "p4"]
    elif encoder in ("mpeg4", "mjpeg"):
        args += ["-q:v", "1"]
    # other codecs (ffv1, png, rawvideo, ...) are lossless already

    return args


def _crop_chunk(video, outputs, extension, codec, crf, lossless, gpu):
    """
    Decode the chunk once and write one cropped video per arena,
    encoded with the same codec as the source chunk
    """
    chunk = os.path.basename(video)[:6]
    encoder_args = get_encoder_args(probe_video(video), codec=codec, crf=crf, lossless=lossless, gpu=gpu)

    filters = [f"[0:v]split={len(outputs)}" + "".join(f"[s{i}]" for i in range(len(outputs)))]
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", video]
    for i, (output, (x0, y0, x1, y1), padding) in enumerate(outputs):
        crop = f"[s{i}]crop={x1 - x0}:{y1 - y0}:{x0}:{y0}"
        if padding is not None:
            width, height, pad_x, pad_y = padding
            crop += f",pad={width}:{height}:{pad_x}:{pad_y}:black"
        filters.append(crop + f"[c{i}]")

    command += ["-filter_complex", ";".join(filters)]
    for i, (output, _, _) in enumerate(outputs):
        command += ["-map", f"[c{i}]", "-an", "-fps_mode", "passthrough"] + encoder_args
        command.append(os.path.join(output, chunk + extension))

    process = subprocess.run(command, stderr=subprocess.PIPE, text=True)
    if process.returncode != 0:
        # e.g. h264_nvenc refuses frames smaller than its minimum dimensions
        raise RuntimeError(f"ffmpeg failed on {video} (exit code {process.returncode}):\n{process.stderr}")

    index = video[:-len(os.path.splitext(video)[1])] + ".npz"
    for output, _, _ in outputs:
        shutil.copy(index, os.path.join(output, chunk + ".npz"))

    return encoder_args


def crop_arenas(input, arenas, output=None, codec=None, crf=12, lossless=False, jobs=4, chunks=None, size=None, gpu=False):
    """
    size: (width, height) of all output videos. ROIs are cropped to their bounding box
    and padded with black up to this size, with the ROI center in the center of the frame.
    If None, each output video has the size of the bounding box of its ROI
    """
    for program in ("ffmpeg", "ffprobe"):
        if shutil.which(program) is None:
            raise RuntimeError(f"{program} not found in PATH")

    fullpath = input if os.path.isfile(input) else get_fullpath(input)
    basedir = os.path.dirname(fullpath)
    with open(fullpath, "r") as filehandle:
        metadata = yaml.load(filehandle, Loader=yaml.SafeLoader)

    store_md = metadata[STORE_MD_KEY]
    imgshape = store_md["imgshape"]
    extension = store_md.get("extension", ".mp4")
    polygons = load_arenas(arenas)

    if output is None:
        output = os.path.join(basedir, "arenas")

    video_chunks = _find_chunks(basedir, extension)
    if chunks is not None:
        video_chunks = {c: v for c, v in video_chunks.items() if c in chunks}
    if not video_chunks:
        raise ValueError(f"No chunks found in {basedir}")

    outputs = []
    for i, polygon in enumerate(polygons):
        bbox = x0, y0, x1, y1 = get_bbox(polygon, imgshape)
        if size is None:
            padding = None
            pad_x = pad_y = 0
            shape = [y1 - y0, x1 - x0]
        else:
            pad_x, pad_y = get_padding(polygon, bbox, size)
            padding = (size[0], size[1], pad_x, pad_y)
            shape = [size[1], size[0]]

        # position in the original frame of the top left corner of the arena video
        offset = np.array([x0 - pad_x, y0 - pad_y])
        folder = os.path.join(output, str(i + 1))
        os.makedirs(folder, exist_ok=True)

        arena_metadata = {k: v for k, v in metadata.items() if k != STORE_MD_KEY}
        arena_metadata[STORE_MD_KEY] = {
            **store_md,
            "imgshape": shape + list(imgshape[2:]),
            "extension": extension,
            "uuid": uuid.uuid4().hex,
        }
        arena_metadata["arena"] = {
            "source": os.path.realpath(basedir),
            "index": i + 1,
            "polygon": polygon.tolist(),
            "bbox": [x0, y0, x1, y1],
            # original coordinates = arena coordinates + offset
            "offset": offset.tolist(),
            "polygon_in_arena": (polygon - offset).tolist(),
        }
        with open(os.path.join(folder, STORE_MD_FILENAME), "w") as filehandle:
            yaml.safe_dump(arena_metadata, filehandle)

        logger.info("Arena %s: x=%s:%s, y=%s:%s (%sx%s) -> %s %sx%s", i + 1, x0, x1, y0, y1, x1 - x0, y1 - y0, folder, shape[1], shape[0])
        outputs.append((folder, bbox, padding))

    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as executor:
        futures = {
            executor.submit(_crop_chunk, video, outputs, extension, codec, crf, lossless, gpu): chunk
            for chunk, video in video_chunks.items()
        }
        for future in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Cropping chunks"):
            encoder_args = future.result()
            logger.debug("Chunk %s encoded with %s", futures[future], " ".join(encoder_args))

    return [folder for folder, _, _ in outputs]


def parse_size(value):
    """
    Parse 1024 or 1024x768 (width x height)
    """
    width, _, height = value.lower().partition("x")
    size = (int(width), int(height or width))
    if any(v <= 0 or v % 2 for v in size):
        raise argparse.ArgumentTypeError(f"--size must be positive and even, got {value}")
    return size


def get_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="Path to the imgstore recording (metadata.yaml or its folder)")
    parser.add_argument("--arenas", required=True, help="Path to the arenas.toml file")
    parser.add_argument("--output", default=None, help="Output folder. Default: arenas/ next to the input metadata.yaml")
    parser.add_argument("--size", type=parse_size, default=None, help="""
        Common size of all arena videos, e.g. 1024 or 1024x1024 (width x height).
        Each ROI is cropped to its bounding box and padded with black to this size, centered.
        Default: the size of each ROI's bounding box""")
    parser.add_argument("--codec", default=None, help="ffmpeg video encoder, e.g. libx264 or h264_nvenc. Default: same codec as the source chunks")
    parser.add_argument("--gpu", action="store_true", default=False, help="Encode h264/hevc chunks on the NVIDIA GPU (h264_nvenc/hevc_nvenc), ~7x faster than libx264")
    parser.add_argument("--crf", type=int, default=12, help="Encoding quality for h264/hevc, lower is better (default 12, visually lossless)")
    parser.add_argument("--lossless", action="store_true", default=False, help="Encode losslessly (h264/hevc only, much bigger files)")
    parser.add_argument("--jobs", type=int, default=4, help="Number of chunks processed in parallel")
    parser.add_argument("--chunks", type=int, nargs="+", default=None, help="Only process these chunks")
    return parser


def main():
    logging.basicConfig(level=logging.INFO)
    args = get_parser().parse_args()
    crop_arenas(args.input, args.arenas, output=args.output, codec=args.codec, crf=args.crf, lossless=args.lossless, jobs=args.jobs, chunks=args.chunks, size=args.size, gpu=args.gpu)


if __name__ == "__main__":
    main()
