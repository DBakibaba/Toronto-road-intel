"""
video_processor.py
------------------
Extracts frames from dashcam clips at a fixed interval
and attaches GPS coordinates to each frame.

Why sample every N seconds instead of every frame?
    275 clips × 1800 frames = 495,000 frames total
    At 0.1s per detection = 13 hours processing time

    275 clips × 30 samples = 8,250 frames total
    At 0.1s per detection = 14 minutes processing time

Depends on: gps_sync.py, gpx_parser.py
"""

import cv2
import os
import pandas as pd
from dataclasses import dataclass
from src.gps_sync import get_gps_for_frame


# ── Configuration ────────────────────────────────────────
SAMPLE_EVERY_N_SECONDS = 2  # extract one frame every 2 seconds
FPS = 30  # your dashcam runs at 30fps
# ─────────────────────────────────────────────────────────


@dataclass
class ExtractedFrame:
    """
    Represents one extracted frame with its metadata.
    A dataclass is like a simple class that just holds data.
    """

    clip_filename: str  # which clip this came from
    frame_number: int  # frame index inside the clip
    timestamp_utc: object  # UTC datetime of this frame
    lat: float  # GPS latitude
    lon: float  # GPS longitude
    elevation: float  # meters above sea level
    image: object  # the actual frame pixels (numpy array)
    interpolated: bool  # was GPS interpolated or exact match


def extract_frames(clip_path: str, gps_df: pd.DataFrame) -> list:
    """
    Extract frames from one clip at regular intervals.
    Attaches GPS coordinates to each frame.

    Parameters
    ----------
    clip_path : str         full path to .MP4 file
    gps_df    : DataFrame   from gpx_parser.parse_gpx()

    Returns
    -------
    list of ExtractedFrame objects
    """

    clip_filename = os.path.basename(clip_path)

    # --- Open the video file ---
    cap = cv2.VideoCapture(clip_path)

    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {clip_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # How many frames between each sample?
    # e.g. 2 seconds × 30 fps = every 60 frames
    sample_interval = int(SAMPLE_EVERY_N_SECONDS * fps)

    extracted = []
    skipped = 0

    # Which frame numbers do we want?
    # range(0, 1800, 60) = [0, 60, 120, 180, ... 1740]
    sample_frames = range(0, total_frames, sample_interval)

    for frame_number in sample_frames:

        # --- Jump directly to this frame (faster than reading all) ---
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        success, image = cap.read()

        if not success:
            skipped += 1
            continue

        # --- Get GPS coordinate for this frame ---
        try:
            gps = get_gps_for_frame(gps_df, clip_filename, frame_number, fps)
        except ValueError:
            # Frame timestamp outside GPS track range — skip it
            skipped += 1
            continue

        extracted.append(
            ExtractedFrame(
                clip_filename=clip_filename,
                frame_number=frame_number,
                timestamp_utc=gps["timestamp"],
                lat=gps["lat"],
                lon=gps["lon"],
                elevation=gps["elevation"],
                image=image,
                interpolated=gps["interpolated"],
            )
        )

    cap.release()

    if skipped > 0:
        print(f"  ⚠️  Skipped {skipped} frames (GPS range or read error)")

    return extracted


def process_clip_summary(frames: list) -> None:
    """
    Print a summary after processing one clip.
    """
    if not frames:
        print("  ❌ No frames extracted")
        return

    print(f"  ✅ Extracted {len(frames)} frames")
    print(
        f"     First: {frames[0].timestamp_utc} "
        f"→ lat={frames[0].lat:.4f}, lon={frames[0].lon:.4f}"
    )
    print(
        f"     Last:  {frames[-1].timestamp_utc} "
        f"→ lat={frames[-1].lat:.4f}, lon={frames[-1].lon:.4f}"
    )
