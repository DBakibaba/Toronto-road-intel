"""
gpx_parser.py
-------------
Reads a GPX file from Open GPX Tracker and returns a clean
pandas DataFrame with UTC timestamps, coordinates, and elevation.

Why UTC everywhere?
    Dashcam filenames use local time (Toronto = UTC-5).
    GPX uses UTC (the Z suffix means UTC).
    We convert dashcam local time → UTC during sync.
    Keeping everything in UTC avoids 5-hour offset bugs.
"""

import gpxpy
import pandas as pd
from datetime import timezone


def parse_gpx(filepath: str) -> pd.DataFrame:
    """
    Parse a GPX file into a clean DataFrame.

    Parameters
    ----------
    filepath : str
        Path to the .gpx file

    Returns
    -------
    pd.DataFrame with columns:
        timestamp   - UTC datetime (timezone-aware)
        lat         - latitude (float)
        lon         - longitude (float)
        elevation   - meters above sea level (float)
    """

    with open(filepath, "r", encoding="utf-8") as f:
        gpx = gpxpy.parse(f)

    points = []

    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:

                # --- Validate point has required data ---
                if point.time is None:
                    continue  # skip points with no timestamp

                if point.latitude is None or point.longitude is None:
                    continue  # skip points with no coordinates

                # --- Ensure timestamp is UTC timezone-aware ---
                # gpxpy returns timestamps as UTC but sometimes
                # without tzinfo attached. We enforce it explicitly.
                ts = point.time.replace(tzinfo=timezone.utc)

                points.append(
                    {
                        "timestamp": ts,
                        "lat": point.latitude,
                        "lon": point.longitude,
                        "elevation": point.elevation if point.elevation else 0.0,
                    }
                )

    if not points:
        raise ValueError(f"No valid GPS points found in: {filepath}")

    df = pd.DataFrame(points)

    # Sort by time — should already be sorted but never assume
    df = df.sort_values("timestamp").reset_index(drop=True)

    return df


def summarize_gpx(df: pd.DataFrame) -> None:
    """
    Print a human-readable summary of a parsed GPX DataFrame.
    Useful for quickly verifying a file loaded correctly.
    """
    duration = df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]
    print("=" * 40)
    print("  GPX TRACK SUMMARY")
    print("=" * 40)
    print(f"  Total points : {len(df)}")
    print(f"  Start (UTC)  : {df['timestamp'].iloc[0]}")
    print(f"  End   (UTC)  : {df['timestamp'].iloc[-1]}")
    print(f"  Duration     : {duration}")
    print(f"  Lat range    : {df['lat'].min():.4f} → {df['lat'].max():.4f}")
    print(f"  Lon range    : {df['lon'].min():.4f} → {df['lon'].max():.4f}")
    print(
        f"  Elevation    : {df['elevation'].min():.1f}m → {df['elevation'].max():.1f}m"
    )
    print("=" * 40)
