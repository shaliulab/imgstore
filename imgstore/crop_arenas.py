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

Arena videos are always square, because imgstore crops non square frames when it reads them.
By default each ROI is cropped to its bounding box, which is padded with white on the bottom or
the right to a square, so the top left corner of the arena video is the one of the bounding box.
Pass --size 1024 to give all arena videos the same size: each ROI is cropped to its
bounding box and padded with white to 1024x1024, with the ROI center in the center
of the frame (the pixel scale is not changed). The arena block of each metadata.yaml
has the offset to map arena coordinates back to the original frame:

    original (x, y) = arena (x, y) + offset

Chunks are re-encoded with the same codec as the source chunks (crf 12 by default, or --lossless)
With --gpu, chunks that fail because the GPU is busy (e.g. no free NVENC session) are retried
an hour later and then whenever the user presses Enter. Other GPU failures are encoded on the CPU

Files that are the same for all arenas (camera .pfs file, chunk .extra.json environment data)
are soft linked into each arena folder, and chunk snapshots (.png) are cropped like the videos.
Pass --no-videos to only do this in arena folders produced before

Pass --resume to continue a run that was killed: chunks already cropped for all arenas are skipped,
and the files of the chunks it was writing (possibly corrupted) are deleted and cropped again.
A chunk is done when every arena folder has its video and a complete copy of its .npz index,
which is only written after all its arena videos are encoded

Once all arenas are cropped, roi-proposer propose (from the roi-proposer environment) is run
on all arena folders to propose their DATE_TIME.toml and landmarks.toml. Pass --no-roi-proposer to skip it

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
import sys
import threading
import time
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

# files of the original recording that are the same for all arenas
SHARED_FILES = ("*.pfs", "[0-9]" * 6 + ".extra.json")
SNAPSHOT_PATTERN = "[0-9]" * 6 + ".png"


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
# CPU encoder used when a GPU encoder cannot be used
CPU_ENCODERS = {encoder: ENCODERS[codec_name] for codec_name, encoder in GPU_ENCODERS.items()}

# ffmpeg errors of a GPU busy with other processes (no free NVENC session or GPU memory)
GPU_BUSY_ERRORS = ("InitializeEncoder failed", "OpenEncodeSessionEx failed", "out of memory", "CUDA_ERROR")
# seconds to wait before retrying a chunk on a busy GPU. If it fails again, the user is asked when to retry
GPU_RETRY_WAIT = 3600

_prompt_lock = threading.Lock()
# number of times the user asked to retry the chunks waiting for the GPU
_retry_requests = 0


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
    stream = probe_video(video)
    encoder_args = get_encoder_args(stream, codec=codec, crf=crf, lossless=lossless, gpu=gpu)

    process = _run_ffmpeg(video, chunk, outputs, extension, encoder_args)
    if process.returncode != 0 and encoder_args[1].endswith("_nvenc"):
        # a GPU shared with other processes may have no free NVENC session (or memory) for a while
        if _gpu_busy(process):
            logger.warning("GPU busy encoding chunk %s, retrying in %s s:\n%s", chunk, GPU_RETRY_WAIT, process.stderr.strip())
            time.sleep(GPU_RETRY_WAIT)
            requests = _retry_requests
            process = _run_ffmpeg(video, chunk, outputs, extension, encoder_args)
            while _gpu_busy(process) and _wait_for_user(chunk, requests):
                requests = _retry_requests
                process = _run_ffmpeg(video, chunk, outputs, extension, encoder_args)

        if process.returncode != 0:
            # e.g. h264_nvenc refuses frames smaller than its minimum dimensions
            cpu_codec = CPU_ENCODERS.get(codec, codec)
            encoder_args = get_encoder_args(stream, codec=cpu_codec, crf=crf, lossless=lossless, gpu=False)
            logger.warning("GPU encoding of chunk %s failed, encoding it with %s:\n%s", chunk, encoder_args[1], process.stderr.strip())
            process = _run_ffmpeg(video, chunk, outputs, extension, encoder_args)

    if process.returncode != 0:
        raise RuntimeError(f"ffmpeg failed on {video} (exit code {process.returncode}):\n{process.stderr}")

    index = _index_path(video)
    for output, _, _ in outputs:
        # copy then rename, so that a complete .npz marks a chunk done for --resume
        dest = os.path.join(output, chunk + ".npz")
        shutil.copy(index, dest + ".part")
        os.replace(dest + ".part", dest)

    return encoder_args


def _index_path(video):
    return video[:-len(os.path.splitext(video)[1])] + ".npz"


def _chunk_files(output, chunk, extension):
    name = os.path.join(output, "%06d" % chunk)
    return name + extension, name + ".npz", name + ".npz.part"


def _chunk_done(chunk, video, outputs, extension):
    """
    True if every arena has the video of the chunk and a complete copy of its .npz index
    (the size is checked too, runs before the .npz copy was atomic may have left a truncated one)
    """
    size = os.path.getsize(_index_path(video))
    for output, _, _ in outputs:
        arena_video, index, _ = _chunk_files(output, chunk, extension)
        if not os.path.exists(arena_video) or not os.path.exists(index) or os.path.getsize(index) != size:
            return False
    return True


def _remove_chunk(chunk, outputs, extension):
    """
    Delete the (possibly partial) files of a chunk in all arenas
    """
    for output, _, _ in outputs:
        for path in _chunk_files(output, chunk, extension):
            if os.path.lexists(path):
                logger.info("Removing incomplete %s", path)
                os.remove(path)


def _gpu_busy(process):
    return process.returncode != 0 and any(error in process.stderr for error in GPU_BUSY_ERRORS)


def _wait_for_user(chunk, requests):
    """
    Wait until the user presses Enter to retry encoding on the GPU.
    One press retries all chunks that failed before it (requests is _retry_requests before their last attempt).
    Return False if there is no user to ask (stdin is not interactive)
    """
    global _retry_requests
    with _prompt_lock:
        if _retry_requests == requests:
            try:
                input(f"\nGPU still busy encoding chunk {chunk}. Press Enter to retry when it is available\n")
            except EOFError:
                return False
            _retry_requests += 1
    return True


def _run_ffmpeg(video, chunk, outputs, extension, encoder_args):
    """
    Decode the chunk once and encode one cropped video per arena with encoder_args
    """
    filters = [f"[0:v]split={len(outputs)}" + "".join(f"[s{i}]" for i in range(len(outputs)))]
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", video]
    for i, (output, (x0, y0, x1, y1), padding) in enumerate(outputs):
        crop = f"[s{i}]crop={x1 - x0}:{y1 - y0}:{x0}:{y0}"
        if padding is not None:
            width, height, pad_x, pad_y = padding
            crop += f",pad={width}:{height}:{pad_x}:{pad_y}:white"
        filters.append(crop + f"[c{i}]")

    command += ["-filter_complex", ";".join(filters)]
    for i, (output, _, _) in enumerate(outputs):
        command += ["-map", f"[c{i}]", "-an", "-fps_mode", "passthrough"] + encoder_args
        command.append(os.path.join(output, chunk + extension))

    return subprocess.run(command, stderr=subprocess.PIPE, text=True)


def link_shared_files(basedir, folder):
    """
    Soft link the files of the original recording that are the same for all arenas
    """
    for pattern in SHARED_FILES:
        for path in sorted(glob.glob(os.path.join(basedir, pattern))):
            dest = os.path.join(folder, os.path.basename(path))
            if os.path.islink(dest):
                os.remove(dest)
            elif os.path.exists(dest):
                logger.warning("%s exists and is not a link. Not replacing it", dest)
                continue
            # relative, so that the links survive moving the whole recording
            os.symlink(os.path.relpath(os.path.realpath(path), os.path.realpath(folder)), dest)


def crop_image(img, bbox, offset, shape):
    """
    Crop an image of the original recording like the arena videos:
    crop to the bbox and pad with white up to shape (height, width)
    """
    x0, y0, x1, y1 = bbox
    pad_x, pad_y = x0 - offset[0], y0 - offset[1]
    cropped = np.full((shape[0], shape[1]) + img.shape[2:], 255, dtype=img.dtype)
    cropped[pad_y:pad_y + y1 - y0, pad_x:pad_x + x1 - x0] = img[y0:y1, x0:x1]
    return cropped


def crop_snapshots(basedir, folder, bbox, offset, shape, chunks=None):
    """
    Write the chunk snapshots (.png) of the original recording cropped to the arena
    """
    for path in sorted(glob.glob(os.path.join(basedir, SNAPSHOT_PATTERN))):
        chunk = int(os.path.basename(path)[:6])
        if chunks is not None and chunk not in chunks:
            continue
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            logger.warning("Cannot read snapshot %s. Skipping", path)
            continue
        cv2.imwrite(os.path.join(folder, os.path.basename(path)), crop_image(img, bbox, offset, shape))


def complete_arena_folder(folder, chunks=None):
    """
    Add to an arena folder the files of the original recording which are not videos:
    soft links to the files shared by all arenas and the cropped chunk snapshots.
    The original recording and the crop are read from the arena metadata.yaml
    """
    with open(os.path.join(folder, STORE_MD_FILENAME), "r") as filehandle:
        metadata = yaml.load(filehandle, Loader=yaml.SafeLoader)

    arena = metadata["arena"]
    basedir = arena["source"]
    link_shared_files(basedir, folder)
    crop_snapshots(basedir, folder, arena["bbox"], arena["offset"], metadata[STORE_MD_KEY]["imgshape"], chunks=chunks)


def crop_arenas(input, arenas, output=None, codec=None, crf=12, lossless=False, jobs=4, chunks=None, size=None, gpu=False, videos=True, resume=False):
    """
    size: (width, height) of all output videos, which must be square. ROIs are cropped to their bounding box
    and padded with white up to this size, with the ROI center in the center of the frame.
    If None, each output video is the bounding box of its ROI, padded with white
    on the bottom or the right to a square
    resume: skip the chunks already cropped for all arenas, and redo the incomplete ones
    """
    for program in ("ffmpeg", "ffprobe"):
        if videos and shutil.which(program) is None:
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

    if not videos:
        # keep the metadata.yaml (and uuid) of the arena folders produced before
        folders = [os.path.join(output, str(i + 1)) for i in range(len(polygons))]
        for folder in folders:
            complete_arena_folder(folder, chunks=chunks)
        return folders

    video_chunks = _find_chunks(basedir, extension)
    if chunks is not None:
        video_chunks = {c: v for c, v in video_chunks.items() if c in chunks}
    if not video_chunks:
        raise ValueError(f"No chunks found in {basedir}")

    outputs = []
    for i, polygon in enumerate(polygons):
        bbox = x0, y0, x1, y1 = get_bbox(polygon, imgshape)
        if size is None:
            # square frames: imgstore crops non square frames when it reads them.
            # Padding the bottom / right keeps the top left corner of the bbox in place
            side = max(x1 - x0, y1 - y0)
            padding = None if (x1 - x0) == (y1 - y0) else (side, side, 0, 0)
            pad_x = pad_y = 0
            shape = [side, side]
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
            # chunk snapshots are optional in arena folders, so the flyhostel dbfile export should not require them
            "must_have_snapshot": False,
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
        metadata_path = os.path.join(folder, STORE_MD_FILENAME)
        if resume and os.path.exists(metadata_path):
            _check_resumable(metadata_path, arena_metadata)
            with open(metadata_path, "r") as filehandle:
                arena_metadata[STORE_MD_KEY]["uuid"] = yaml.load(filehandle, Loader=yaml.SafeLoader)[STORE_MD_KEY]["uuid"]

        with open(os.path.join(folder, STORE_MD_FILENAME), "w") as filehandle:
            yaml.safe_dump(arena_metadata, filehandle)

        logger.info("Arena %s: x=%s:%s, y=%s:%s (%sx%s) -> %s %sx%s", i + 1, x0, x1, y0, y1, x1 - x0, y1 - y0, folder, shape[1], shape[0])
        outputs.append((folder, bbox, padding))

    if resume:
        done = [chunk for chunk, video in video_chunks.items() if _chunk_done(chunk, video, outputs, extension)]
        for chunk in done:
            del video_chunks[chunk]
        for chunk in video_chunks:
            _remove_chunk(chunk, outputs, extension)
        logger.info("Resuming: %s chunks already cropped, %s to crop", len(done), len(video_chunks))

    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as executor:
        futures = {
            executor.submit(_crop_chunk, video, outputs, extension, codec, crf, lossless, gpu): chunk
            for chunk, video in video_chunks.items()
        }
        for future in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Cropping chunks"):
            encoder_args = future.result()
            logger.debug("Chunk %s encoded with %s", futures[future], " ".join(encoder_args))

    for folder, _, _ in outputs:
        complete_arena_folder(folder, chunks=chunks)

    return [folder for folder, _, _ in outputs]


def _check_resumable(metadata_path, arena_metadata):
    """
    Refuse to resume into an arena folder produced with a different crop (arenas, --size or input)
    """
    with open(metadata_path, "r") as filehandle:
        previous = yaml.load(filehandle, Loader=yaml.SafeLoader)
    for key in ("source", "bbox", "offset"):
        if previous.get("arena", {}).get(key) != arena_metadata["arena"][key]:
            raise ValueError(f"Cannot resume: {key} of {metadata_path} differs from this run. Run without --resume to start from scratch")
    if previous[STORE_MD_KEY].get("imgshape") != arena_metadata[STORE_MD_KEY]["imgshape"]:
        raise ValueError(f"Cannot resume: imgshape of {metadata_path} differs from this run. Run without --resume to start from scratch")


def find_roi_proposer():
    """roi-proposer lives in the roi-proposer environment, next to this one (PATH is only a fallback,
    other environments ship an older roi-proposer)"""
    envs = [os.path.dirname(sys.prefix)]
    if os.environ.get("CONDA_EXE"):
        envs.append(os.path.join(os.path.dirname(os.path.dirname(os.environ["CONDA_EXE"])), "envs"))
    for env_dir in envs:
        path = os.path.join(env_dir, "roi-proposer", "bin", "roi-proposer")
        if os.path.exists(path):
            return path
    if shutil.which("roi-proposer"):
        return shutil.which("roi-proposer")
    raise FileNotFoundError("roi-proposer not found in the roi-proposer environment, pass --roi-proposer")


def propose_rois(folders, executable=None):
    """
    Propose the idtrackerai tomls of every arena folder with roi-proposer propose
    """
    if executable is None:
        executable = find_roi_proposer()
    logger.info("Proposing ROIs with %s propose for %s arenas", executable, len(folders))
    process = subprocess.run([executable, "propose"] + list(folders))
    if process.returncode != 0:
        logger.error("roi-proposer propose failed on some arenas (exit code %s)", process.returncode)
    return process.returncode


def parse_size(value):
    """
    Parse 1024 or 1024x1024 (width x height)
    """
    width, _, height = value.lower().partition("x")
    size = (int(width), int(height or width))
    if any(v <= 0 or v % 2 for v in size):
        raise argparse.ArgumentTypeError(f"--size must be positive and even, got {value}")
    if size[0] != size[1]:
        raise argparse.ArgumentTypeError(f"--size must be square (imgstore crops non square frames), got {value}")
    return size


def get_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="Path to the imgstore recording (metadata.yaml or its folder)")
    parser.add_argument("--arenas", required=True, help="Path to the arenas.toml file")
    parser.add_argument("--output", default=None, help="Output folder. Default: arenas/ next to the input metadata.yaml")
    parser.add_argument("--size", type=parse_size, default=None, help="""
        Common size of all arena videos, e.g. 1024 or 1024x1024 (width x height, must be square).
        Each ROI is cropped to its bounding box and padded with white to this size, centered.
        Default: each ROI's bounding box, padded with white on the bottom or the right to a square""")
    parser.add_argument("--codec", default=None, help="ffmpeg video encoder, e.g. libx264 or h264_nvenc. Default: same codec as the source chunks")
    parser.add_argument("--gpu", action="store_true", default=False, help="Encode h264/hevc chunks on the NVIDIA GPU (h264_nvenc/hevc_nvenc), ~7x faster than libx264")
    parser.add_argument("--crf", type=int, default=12, help="Encoding quality for h264/hevc, lower is better (default 12, visually lossless)")
    parser.add_argument("--lossless", action="store_true", default=False, help="Encode losslessly (h264/hevc only, much bigger files)")
    parser.add_argument("--jobs", type=int, default=4, help="Number of chunks processed in parallel")
    parser.add_argument("--chunks", type=int, nargs="+", default=None, help="Only process these chunks")
    parser.add_argument("--no-videos", dest="videos", action="store_false", default=True, help="""
        Do not crop the videos. Only link the files shared by all arenas and crop the snapshots
        in the arena folders produced before""")
    parser.add_argument("--resume", action="store_true", default=False, help="""
        Continue a killed run: skip the chunks already cropped for all arenas,
        delete the files of the incomplete ones and crop them again""")
    parser.add_argument("--roi-proposer", default=None, help="roi-proposer executable (default: the one in the roi-proposer environment)")
    parser.add_argument("--no-roi-proposer", dest="propose_rois", action="store_false", default=True, help="""
        Do not run roi-proposer propose on the arena folders after cropping them""")
    return parser


def main():
    logging.basicConfig(level=logging.INFO)
    args = get_parser().parse_args()
    folders = crop_arenas(args.input, args.arenas, output=args.output, codec=args.codec, crf=args.crf, lossless=args.lossless, jobs=args.jobs, chunks=args.chunks, size=args.size, gpu=args.gpu, videos=args.videos, resume=args.resume)
    if args.propose_rois:
        return propose_rois(folders, executable=args.roi_proposer)


if __name__ == "__main__":
    main()
