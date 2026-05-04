"""
map_visualizer.py
-----------------
Reads pothole detections from SQLite and builds an interactive
Folium map of Toronto road damage hotspots.

Usage:
    python src/map_visualizer.py
    python src/map_visualizer.py --output output/toronto_potholes.html
    python src/map_visualizer.py --min-confidence 0.5
"""

import sqlite3
import argparse
import os
import folium
from folium.plugins import HeatMap, MarkerCluster

# ── Config ────────────────────────────────────────────────────────────────────
DB_PATH = "data/detections/road_intel.db"
OUTPUT_PATH = "output/toronto_potholes.html"

# Toronto city center — map starts here
MAP_CENTER = [43.7, -79.42]
MAP_ZOOM = 12


# ── Database ──────────────────────────────────────────────────────────────────
def load_detections(db_path: str, min_confidence: float = 0.0) -> list[dict]:
    """
    Load all detection from SQlite.
    Returns a list of dicts,one per detection

    """

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    cursor = conn.cursor()
    cursor.execute(
        """
                   SELECT id,damage_type,confidence,lat,lon,timestamp_utc,clip_filename,frame_number
                   FROM detections
                   WHERE lat Is NOT NULL
                   AND lon IS NOT NULL
                   AND confidence >=? 
                   ORDER BY confidence DESC
                   """,
        (min_confidence,),
    )
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows


# ── Color ─────────────────────────────────────────────────────────────────────
def confidence_color(confidence: float) -> str:
    """
    Map confidence score to a marker color.
    Red = high confidence (real pothole)
    Orange = medium
    Green = low confidence (possible false positive)
    """
    if confidence >= 0.7:
        return "red"
    elif confidence >= 0.4:
        return "orange"
    else:
        return "green"


def get_image_path(clip_filename: str, frame_number: int) -> str | None:
    if not clip_filename:
        return None

    stem = os.path.splitext(clip_filename)[0]

    # Try full filename first
    path = f"output/raw_frames/{stem}_f{frame_number:04d}.jpg"
    if os.path.exists(path):
        return path

    # Try shortened format — "NO20260223-083804-001173F" → "NO20260223-0838"
    # Take first 16 characters: "NO20260223-0838"
    short_stem = stem[:15]
    short_path = f"output/raw_frames/{short_stem}_f{frame_number:04d}.jpg"
    if os.path.exists(short_path):
        return short_path

    return None


# ── Map ───────────────────────────────────────────────────────────────────────
def build_map(detections: list[dict]) -> folium.Map:
    """
    Build and return a Folium map with:
    - Individual markers (clustered) showing each detection
    - Heatmap layer showing density hotspots
    - Layer control to toggle between views
    """
    m = folium.Map(
        location=MAP_CENTER,
        zoom_start=MAP_ZOOM,
        tiles="CartoDB positron",  # clean light basemap, good for road data
    )

    # ── Marker cluster layer ──────────────────────────────────────────────────
    # MarkerCluster groups nearby dots into a number bubble at low zoom
    # and spreads them out as you zoom in — prevents 1,091 overlapping pins
    cluster = MarkerCluster(name="Individual Detections").add_to(m)

    for d in detections:
        # Build popup text shown when user clicks a marker
        # Check if image exists for this detection
        img_path = get_image_path(d["clip_filename"], d["frame_number"])

        if img_path:
            # Convert image to base64 so it embeds directly in HTML
            import base64

            with open(img_path, "rb") as f:
                img_data = base64.b64encode(f.read()).decode("utf-8")
            img_tag = f'<img src="data:image/jpeg;base64,{img_data}" width="250px"><br>'
        else:
            img_tag = "<i>No image available</i><br>"

        popup_html = f"""
        <div style="font-family: Arial; font-size: 13px; width: 260px;">
            <b>🕳️ {d['damage_type']}</b><br>
            {img_tag}
            <hr style="margin: 4px 0">
            <b>Confidence:</b> {d['confidence']:.2f}<br>
            <b>Date:</b> {d['timestamp_utc'][:10] if d['timestamp_utc'] else 'N/A'}<br>
            <b>Clip:</b> {d['clip_filename']}<br>
            <b>GPS:</b> {d['lat']:.5f}, {d['lon']:.5f}
        </div>
        """

        folium.CircleMarker(
            location=[d["lat"], d["lon"]],
            radius=6,
            color=confidence_color(d["confidence"]),
            fill=True,
            fill_color=confidence_color(d["confidence"]),
            fill_opacity=0.8,
            popup=folium.Popup(popup_html, max_width=240),
            tooltip=f"Confidence: {d['confidence']:.2f}",
        ).add_to(cluster)

    # ── Heatmap layer ─────────────────────────────────────────────────────────
    # Shows density — where are the most potholes concentrated?
    # Each point is weighted by its confidence score
    heat_data = [[d["lat"], d["lon"], d["confidence"]] for d in detections]

    HeatMap(
        heat_data,
        name="Heatmap",
        min_opacity=0.3,
        radius=18,
        blur=15,
        gradient={0.4: "blue", 0.6: "lime", 0.8: "orange", 1.0: "red"},
    ).add_to(m)

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_html = """
    <div style="
        position: fixed; bottom: 40px; left: 40px; z-index: 1000;
        background: white; padding: 12px 16px; border-radius: 8px;
        border: 1px solid #ccc; font-family: Arial; font-size: 13px;
        box-shadow: 2px 2px 6px rgba(0,0,0,0.2);">
        <b>🕳️ Toronto Road Intel</b><br>
        <i style="color:red">●</i> High confidence (&gt;0.70)<br>
        <i style="color:orange">●</i> Medium confidence (0.40–0.70)<br>
        <i style="color:green">●</i> Low confidence (&lt;0.40)<br>
        <hr style="margin: 6px 0">
        <small>{} detections total</small>
    </div>
    """.format(len(detections))

    m.get_root().html.add_child(folium.Element(legend_html))

    # ── Layer control ─────────────────────────────────────────────────────────
    # Toggle between markers and heatmap in top-right corner of map
    folium.LayerControl().add_to(m)

    return m


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Build Toronto pothole map")
    parser.add_argument("--output", default=OUTPUT_PATH)
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.0,
        help="Only show detections above this confidence (0.0-1.0)",
    )
    args = parser.parse_args()

    # Load
    print(f"Loading detections from {DB_PATH}...")
    detections = load_detections(DB_PATH, args.min_confidence)
    print(f"Loaded {len(detections)} detections")

    if not detections:
        print("No detections found. Check your DB path or confidence threshold.")
        return

    # Build
    print("Building map...")
    m = build_map(detections)

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    m.save(args.output)
    print(f"Map saved → {args.output}")
    print(f"Open in browser: file:///{os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
