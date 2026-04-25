# check_db.py
import sqlite3

conn = sqlite3.connect("data/detections/road_intel.db")
cursor = conn.cursor()

cursor.execute(
    """
    SELECT substr(clip_filename, 1, 10), COUNT(*) 
    FROM detections 
    GROUP BY substr(clip_filename, 1, 10)
"""
)

rows = cursor.fetchall()
for row in rows:
    print(f"Date: {row[0]}  |  Detections: {row[1]}")

conn.close()
