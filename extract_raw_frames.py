"""
extract_raw_frames.py
---------------------
Reads detections from database and extracts clean
raw frames from original MP4 files for Roboflow labeling

Usage:
    python exact_raw_frames.py

"""

import cv2
import os
import sqlite3

DB_PATH = "data/detections/road_intel.db"
VIDEO_FOLDER = "data/raw_video"
OUTPUT_FOLDER = "output/raw_frames"


def extract_raw_frame(video_path, frame_number, output_path):
    """
    Opens a video, seek to exact frame, save clean image.
    Returns True if succesful, False if failed.

    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return False
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    success, frame = cap.read()

    if success:
        cv2.imwrite(output_path, frame)
    cap.release()
    return success


def find_video_path(clip_filename):
    """
    Find the full path to an MP4 file by searching all date folders.
    clip_filename looks like: NO20260422-1136-000001F.MP4

    """
    # Extract date from filename — first 8 digits after 'NO'
    # NO20260422 → 20260422

    date = clip_filename[2:10]
    full_path = os.path.join(VIDEO_FOLDER, date, clip_filename)

    if os.path.exists(full_path):
        return full_path

    return None


def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM detections")
    detections = cursor.fetchall()
    conn.close()

    print(f"Found {len(detections)} detections in database")
    print(f"Extracting raw frames to {OUTPUT_FOLDER}/\n")

    success_count = 0
    fail_count = 0

    for i, detection in enumerate(detections):
        clip_filename = detection["clip_filename"]
        frame_number = detection["frame_number"]

        # Build output filename to match annotated frames
        out_name = f"{clip_filename[:15]}_f{frame_number:04d}.jpg"
        output_path = os.path.join(OUTPUT_FOLDER, out_name)

        # Skip if already extracted
        if os.path.exists(output_path):
            success_count += 1
            continue
        # Find the video file
        video_path = find_video_path(clip_filename)

        if video_path is None:
            print(f"Video not found:{clip_filename}")
            fail_count += 1
            continue
        # Extract the frame
        ok = extract_raw_frame(video_path, frame_number, output_path)

        if ok:
            success_count += 1
        else:
            print(f"  ❌ Frame read failed: {clip_filename} frame {frame_number}")
            fail_count += 1

        # Progress every 50 frames
        if (i + 1) % 50 == 0:
            print(f"  Progress: {i+1}/{len(detections)}")

    print(f"\n{'='*40}")
    print(f"  Done!")
    print(f"  Extracted : {success_count}")
    print(f"  Failed    : {fail_count}")
    print(f"  Output    : {OUTPUT_FOLDER}/")
    print(f"{'='*40}")


if __name__ == "__main__":
    main()
