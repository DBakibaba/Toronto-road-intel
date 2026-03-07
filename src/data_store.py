"""
data_store.py
-------------
Handles all database operations for Toronto Road Intel.
Saves detections to a SQLite database and retrieves them for analysis.

Database location: data/detections/road_intel.db
"""

import sqlite3
import os
from src.detection_engine import Detection

DB_PATH = "data/detections/road_intel.db"


def create_table() -> None:
    """
    Create the detections table if it doesn't exist yet.
    Safe to call every time the program starts.

    """

    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute(
        """
            CREATE TABLE IF NOT EXISTS detections
                (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                damage_type TEXT,
                confidence REAL,
                lat REAL,
                lon REAL,
                elevation REAL,
                timestamp_utc TEXT,
                clip_filename TEXT,
                frame_number INTEGER,
                interpolated INTEGER
            )
                  
                   
         """
    )

    conn.commit()
    conn.close()


def save_detection(detection: Detection) -> None:
    """
    Save one detection to the database.
    Called evert time the detection engine finds a pothole.

    """

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute(
        """
    INSERT INTO detections(
                   damage_type,confidence,lat,lon,elevation,timestamp_utc,clip_filename,frame_number,interpolated) 
                   VALUES(?,?,?,?,?,?,?,?,?)
        """,
        (
            detection.damage_type,
            detection.confidence,
            detection.lat,
            detection.lon,
            detection.elevation,
            str(detection.timestamp_utc),
            detection.clip_filename,
            detection.frame_number,
            int(detection.interpolated),
        ),
    )
    conn.commit()
    conn.close()

    
def get_all_detection()->list:
    """ 
    Retrieve all detection from the database.
    Returns a list of dictionaries- one per detection.
    Used for dashboard export and analysis.

    """

    conn=sqlite3.connect(DB_PATH)
    #access by column name
    conn.row_factory=sqlite3.Row
    cursor=conn.cursor()

    cursor.execute("SELECT * FROM detections")
    rows=cursor.fetchall()

    conn.close()

    return[dict(row) for row in rows]