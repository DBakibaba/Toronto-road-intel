"""
gps_sync.py
-----------
Answers one question:
"Given a dashcam clip filename and a frame number,
what GPS coordinate was I at?"

Depends on: gpx_parser.py (must be run first)
"""

import re
from datetime import datetime, timezone, timedelta
import pandas as pd


# Toronto winter time is UTC-5 (EST)
# Note: hardcoded for winter operation — daylight saving (UTC-4) not yet implemented

TORONTO_UTC_OFFSET = timedelta(hours=5)


def parse_clip_start_utc(filename: str) -> datetime:
    """
    Extract the start time from a dashcam filename and
    convert from Toronto local time to UTC.

    Example:
        NO20260223-083304-001168F.MP4
              ↑        ↑
              date     time (local Toronto)

        Returns: 2026-02-23 13:33:04 UTC
    """

    # --- Extract date and time from filename using regex ---
    # Pattern: NO + YYYYMMDD + - + HHMMSS
    pattern = r"NO(\d{8})-(\d{6})-"
    match = re.search(pattern, filename)

    if not match:
        raise ValueError(
            f"Filename doesn't match expected pattern: {filename}\n"
            f"Expected format: NO20260223-083304-001168F.MP4"
        )

    date_str = match.group(1)  # "20260223"
    time_str = match.group(2)  # "083304"

    # --- Parse into datetime object ---
    # We tell Python this is a naive local time first
    local_time = datetime.strptime(date_str + time_str, "%Y%m%d%H%M%S")

    # --- Convert Toronto local → UTC ---
    # Toronto winter time = UTC - 5
    # So UTC = local + 5 hours
    utc_time = local_time + TORONTO_UTC_OFFSET

    # --- Attach UTC timezone info ---
    utc_time = utc_time.replace(tzinfo=timezone.utc)

    return utc_time


def get_frame_timestamp(
    clip_filename: str, frame_number: int, fps: float = 30.0
) -> datetime:
    """
    Calculate the exact UTC timestamp of a specific frame.

    Parameters
    ----------
    clip_filename : str   e.g. "NO20260223-083304-001168F.MP4"
    frame_number  : int   e.g. 450
    fps           : float e.g. 30.0

    Returns
    -------
    datetime (UTC, timezone-aware)
    """

    clip_start = parse_clip_start_utc(clip_filename)

    # How many seconds into the clip is this frame?
    seconds_offset = frame_number / fps

    frame_time = clip_start + timedelta(seconds=seconds_offset)

    return frame_time


def interpolate_gps(df: pd.DataFrame, target_time: datetime) -> dict:
    """
    Find the GPS coordinate at a specific UTC timestamp
    by interpolating between the two nearest GPS points.

    Parameters
    ----------
    df          : DataFrame from gpx_parser.parse_gpx()
    target_time : UTC datetime we want coordinates for

    Returns
    -------
    dict with keys: lat, lon, elevation, timestamp, interpolated
    """

    # --- Check target is within our GPS track range ---
    track_start = df["timestamp"].iloc[0]
    track_end = df["timestamp"].iloc[-1]

    if target_time < track_start or target_time > track_end:
        raise ValueError(
            f"Target time {target_time} is outside GPS track range.\n"
            f"Track runs from {track_start} to {track_end}"
        )

    # --- Find the GPS point just BEFORE target time ---
    before = df[df["timestamp"] <= target_time]
    after = df[df["timestamp"] > target_time]

    # --- If exact match exists, return it directly ---
    if before.iloc[-1]["timestamp"] == target_time:
        row = before.iloc[-1]
        return {
            "lat": row["lat"],
            "lon": row["lon"],
            "elevation": row["elevation"],
            "timestamp": target_time,
            "interpolated": False,  # exact match, no estimation
        }

    point_before = before.iloc[-1]  # last point before target
    point_after = after.iloc[0]  # first point after target

    # --- Calculate interpolation fraction ---
    # How far between the two points is our target time?
    #
    # Example:
    #   before = 13:33:17,  after = 13:33:22,  target = 13:33:19
    #   total_gap   = 5 seconds
    #   time_into   = 2 seconds
    #   fraction    = 2/5 = 0.4

    total_gap = (point_after["timestamp"] - point_before["timestamp"]).total_seconds()

    time_into = (target_time - point_before["timestamp"]).total_seconds()

    fraction = time_into / total_gap

    # --- Interpolate lat, lon, elevation ---
    lat = point_before["lat"] + fraction * (point_after["lat"] - point_before["lat"])

    lon = point_before["lon"] + fraction * (point_after["lon"] - point_before["lon"])

    ele = point_before["elevation"] + fraction * (
        point_after["elevation"] - point_before["elevation"]
    )

    return {
        "lat": lat,
        "lon": lon,
        "elevation": ele,
        "timestamp": target_time,
        "interpolated": True,  # estimated between two points
    }


def get_gps_for_frame(
    df: pd.DataFrame, clip_filename: str, frame_number: int, fps: float = 30.0
) -> dict:
    """
    Master function — combines everything above.
    Given a clip and frame number, returns GPS coordinates.

    This is the function the detection engine will call.
    """

    frame_time = get_frame_timestamp(clip_filename, frame_number, fps)

    gps = interpolate_gps(df, frame_time)

    return gps
