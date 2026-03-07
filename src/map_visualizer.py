"""
map_visualizer.py
-----------------
Planned module for rendering detected potholes on an interactive map.

Intended approach:
    - Load detections from SQLite database (via data_store.py)
    - Plot GPS coordinates on a Folium or Plotly map
    - Color-code by confidence score (high / medium / low)
    - Export as interactive HTML map

Current status: In development — data pipeline and storage layer
being completed first.

Depends on: data_store.py
"""
