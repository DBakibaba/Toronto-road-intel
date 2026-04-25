"""
extract_all_frames.py
---------------------
Extracts every frame at 2 second intervals from all MP4 clips
for a given date, regardless of detection.

Usage:
    python extract_all_frames.py --date 20260422
"""

import cv2
import os
import argparse

VIDEO_FOLDER = "data/raw_video"
OUTPUT_FOLDER = "output/all_frames"
INTERVAL_SECONDS = 2


def extract_frames_from_clip(clip_path, output_folder):
    cap = cv2.VideoCapture(clip_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * INTERVAL_SECONDS)

    frame_count = 0
    saved_count = 0
    clip_name = os.path.basename(clip_path)

    while True:
        success, frame = cap.read()

        if not success:
            break  # video ended, exit loop cleanly

        if frame_count % frame_interval == 0:
            out_name = f"{clip_name[:15]}_f{frame_count:04d}.jpg"
            out_path = os.path.join(output_folder, out_name)
            cv2.imwrite(out_path, frame)
            saved_count += 1

        frame_count += 1

    cap.release()
    return saved_count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True)
    args = parser.parse_args()

    video_folder = os.path.join(VIDEO_FOLDER, args.date)
    clips = sorted(
        [
            os.path.join(video_folder, f)
            for f in os.listdir(video_folder)
            if f.endswith(".MP4")
        ]
    )

    output_folder = os.path.join(OUTPUT_FOLDER, args.date)
    os.makedirs(output_folder, exist_ok=True)

    print(f"Found {len(clips)} clips for {args.date}")

    total_saved = 0
    for i, clip in enumerate(clips):
        count = extract_frames_from_clip(clip, output_folder)
        total_saved += count
        print(f"[{i+1}/{len(clips)}] {os.path.basename(clip)} → {count} frames")

    print(f"\nTotal frames extracted: {total_saved}")
    print(f"Saved to: {output_folder}")


if __name__ == "__main__":
    main()
