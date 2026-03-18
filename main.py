"""
main.py
-------
Entry point for Toronto Road Intel pipeline.

Usage:
    python main.py --date 20260305
    python main.py --date 20260305 --clips 10
    python main.py --date 20260305 --clips all
"""

import argparse
import os
import cv2
import glob
from src.gpx_parser import parse_gpx, summarize_gpx
from src.video_processor import extract_frames
from src.detection_engine import load_model, process_frame, interactive_crop
from src.data_store import create_table, save_detection


# ── Configuration ─────────────────────────────────────────
GPX_FOLDER = "data/gps_tracks"
VIDEO_FOLDER = "data/raw_video"
OUTPUT_FOLDER = "output/annotated_frames"
# ──────────────────────────────────────────────────────────


def find_gpx_for_date(date: str) -> str:
    """
    Find the GPX file that matches a given date (YYYYMMDD).
    e.g. 20260305 → matches '05-Mar-2026-1005.gpx'
    """
    # Convert YYYYMMDD → day, month, year parts for matching
    year = date[0:4]
    month = date[4:6]
    day = date[6:8]

    # Month number to abbreviation
    months = {
        "01": "Jan",
        "02": "Feb",
        "03": "Mar",
        "04": "Apr",
        "05": "May",
        "06": "Jun",
        "07": "Jul",
        "08": "Aug",
        "09": "Sep",
        "10": "Oct",
        "11": "Nov",
        "12": "Dec",
    }
    month_abbr = months[month]

    # Match pattern like "05-Mar-2026"
    prefix = f"{day}-{month_abbr}-{year}"

    all_gpx = os.listdir(GPX_FOLDER)
    for f in all_gpx:
        if f.startswith(prefix):
            return os.path.join(GPX_FOLDER, f)

    raise FileNotFoundError(
        f"No GPX file found for date {date}\n"
        f"Looking for file starting with: {prefix}\n"
        f"Available files: {all_gpx}"
    )


def find_clips_for_date(date: str) -> list:
    """
    Get all video clips for a given date, sorted by time.
    """
    folder = os.path.join(VIDEO_FOLDER, date)

    if not os.path.exists(folder):
        raise FileNotFoundError(
            f"No video folder found for date {date}\n"
            f"Expected: {folder}\n"
            f"Run: bash scripts/import_shift.sh {date}"
        )

    clips = sorted(
        [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".MP4")]
    )

    return clips


def run_pipeline(date: str, num_clips: int = 10):
    """
    Run the full detection pipeline for one shift.
    """
    print("=" * 50)
    print("  TORONTO ROAD INTEL")
    create_table()
    print(f"  Date: {date}")
    print("=" * 50)

    # ── Step 1: Load GPS ──────────────────────────────────
    print("\n📍 Loading GPS data...")
    gpx_path = find_gpx_for_date(date)
    print(f"   File: {gpx_path}")
    df = parse_gpx(gpx_path)
    summarize_gpx(df)

    # ── Step 2: Load Model ────────────────────────────────
    print("\n🤖 Loading detection model...")
    model = load_model()

    # ── Step 3: Get Clips ─────────────────────────────────
    print(f"\n🎬 Loading clips for {date}...")
    all_clips = find_clips_for_date(date)
    print(f"   Total available: {len(all_clips)} clips")

    if num_clips == -1:  # -1 means all
        clips_to_process = all_clips
    else:
        clips_to_process = all_clips[5 : 5 + num_clips]

    print(f"   Processing: {len(clips_to_process)} clips")
    print(f"   From: {os.path.basename(clips_to_process[0])}")
    print(f"   To:   {os.path.basename(clips_to_process[-1])}")

    # ── Step 4: Run Detection ─────────────────────────────
    print(f"\n🔍 Running detection...")
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # Clear old frames
    for f in glob.glob(f"{OUTPUT_FOLDER}/*.jpg"):
        os.remove(f)

    total_detections = 0
    total_frames_scanned = 0

    for i, clip_path in enumerate(clips_to_process):
        clip_name = os.path.basename(clip_path)
        print(f"\n  [{i+1}/{len(clips_to_process)}] {clip_name}")

        try:
            frames = extract_frames(clip_path, df)
        except Exception as e:
            print(f"  ⚠️  Skipped: {e}")
            continue

        total_frames_scanned += len(frames)

        crop_values = interactive_crop(frames[0].image)

        for extracted_frame in frames:
            detections = process_frame(model, extracted_frame, crop_values)

            if not detections:
                continue  # ← this works now because we're inside a loo
            # Draw boxes on frame
            img = extracted_frame.image.copy()
            for d in detections:
                x1, y1, x2, y2 = [int(v) for v in d.bbox]
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(
                    img,
                    f"Pothole {d.confidence:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                )

            # Save annotated frame
            out_name = f"{clip_name[:15]}_f{extracted_frame.frame_number:04d}.jpg"
            cv2.imwrite(os.path.join(OUTPUT_FOLDER, out_name), img)

            total_detections += len(detections)
            for d in detections:
                save_detection(d)
                print(
                    f"  🕳️  POTHOLE | frame {extracted_frame.frame_number} "
                    f"| conf={d.confidence:.2f} "
                    f"| lat={d.lat:.5f} | lon={d.lon:.5f}"
                )

    # ── Step 5: Summary ───────────────────────────────────
    print("\n" + "=" * 50)
    print("  DETECTION COMPLETE")
    print(f"  Clips processed : {len(clips_to_process)}")
    print(f"  Frames scanned  : {total_frames_scanned}")
    print(f"  Potholes found  : {total_detections}")
    print(f"  Saved frames    : {OUTPUT_FOLDER}/")
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Toronto Road Intel Pipeline")

    parser.add_argument(
        "--date", required=True, help="Shift date in YYYYMMDD format (e.g. 20260305)"
    )
    parser.add_argument(
        "--clips",
        default="10",
        help="Number of clips to process, or 'all' (default: 10)",
    )

    args = parser.parse_args()

    # Parse clips argument
    if args.clips == "all":
        num_clips = -1
    else:
        num_clips = int(args.clips)

    run_pipeline(date=args.date, num_clips=num_clips)
