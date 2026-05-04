import os
import sqlite3

conn = sqlite3.connect("data/detections/road_intel.db")
conn.row_factory = sqlite3.Row
cursor = conn.cursor()
cursor.execute("SELECT clip_filename, frame_number FROM detections LIMIT 3")
rows = cursor.fetchall()

for r in rows:
    stem = os.path.splitext(r["clip_filename"])[0]
    path = f"output/raw_frames/{stem}_f{r['frame_number']:04d}.jpg"
    print(f"DB clip_filename : {r['clip_filename']}")
    print(f"DB frame_number  : {r['frame_number']}")
    print(f"Looking for      : {path}")
    print(f"Exists           : {os.path.exists(path)}")
    print()

# Also show what actually exists in raw_frames
print("Files actually in output/raw_frames/:")
files = os.listdir("output/raw_frames")[:5]
for f in files:
    print(f"  {f}")
print("\nFeb 23 detections in DB:")
cursor.execute(
    "SELECT clip_filename, frame_number FROM detections WHERE clip_filename LIKE 'NO20260223%' LIMIT 3"
)
for r in cursor.fetchall():
    stem = os.path.splitext(r["clip_filename"])[0]
    path = f"output/raw_frames/{stem}_f{r['frame_number']:04d}.jpg"
    print(f"Looking for: {path}")
    print(f"Exists: {os.path.exists(path)}")
