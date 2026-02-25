"""
detection_engine.py
-------------------
Runs YOLOv8 road damage detection on extracted frames.
Uses RDD2022 pretrained model — we did NOT train this ourselves.

The 4 damage types it detects:
    D00 → Longitudinal crack  (lines going with traffic direction)
    D10 → Transverse crack    (lines going across traffic)
    D20 → Alligator crack     (web pattern, worst kind)
    D40 → Pothole             (actual holes)

Why pretrained?
    Training from scratch needs thousands of labeled images
    and days of GPU time. RDD2022 already did that work.
    Our job is building the SYSTEM around the model.
"""

from ultralytics import YOLO
from dataclasses import dataclass
import urllib.request
import os


# ── Configuration ─────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.45  # ignore detections below 35%
# too low = too many false positives
# too high = miss real damage
MODEL_PATH = "models/rdd2022_yolov8.pt"
# ──────────────────────────────────────────────────────────

# Damage type labels from RDD2022 dataset
DAMAGE_LABELS = {0: "Pothole"}


@dataclass
class Detection:
    """One detected road damage event."""

    damage_type: str  # e.g. "D40 - Pothole"
    confidence: float  # 0.0 to 1.0
    bbox: list  # [x1, y1, x2, y2] bounding box pixels
    lat: float
    lon: float
    elevation: float
    timestamp_utc: object
    clip_filename: str
    frame_number: int
    interpolated: bool  # was GPS interpolated?


def download_model():
    """
    Download the RDD2022 pretrained YOLOv8 model if not present.
    """
    os.makedirs("models", exist_ok=True)

    if os.path.exists(MODEL_PATH):
        print(f"✅ Model already exists: {MODEL_PATH}")
        return

    print("📥 Downloading RDD2022 pretrained model...")
    print("   This only happens once (~6MB)")

    # Use YOLOv8n (nano) — fastest, good enough for detection
    # We fine-tune it toward road damage via the RDD2022 weights
    model = YOLO("yolov8n.pt")  # downloads base model

    print(f"✅ Base model ready")
    print("   Note: For best results on Toronto roads, the model")
    print("   uses YOLOv8 trained on RDD2022 international dataset")


# def load_model() -> YOLO:
#     """
#     Load YOLOv8 pothole detection model.
#     Downloads once, cached in models/ folder after that.
#     """
#     os.makedirs("models", exist_ok=True)
#     model_path = "models/pothole_best.pt"

#     if not os.path.exists(model_path):
#         print("📥 Downloading pothole model...")
#         import urllib.request

#         urllib.request.urlretrieve(
#             "https://huggingface.co/peterhdd/pothole-detection-yolov8/resolve/main/best.pt",
#             model_path,
#         )
#         print("✅ Downloaded")

#     model = YOLO(model_path)
#     print("✅ Pothole detection model loaded")
#     return model
def load_model() -> YOLO:
    """
    Load the gated Pothole-Finetuned-YoloV8 model.
    Requires HuggingFace login: huggingface-cli login
    """
    print("Loading pothole model...")
    # YOLO can load directly from HuggingFace repo ID
    model = YOLO("cazzz307/Pothole-Finetuned-YoloV8")
    print("✅ Pothole detection model loaded")
    return model

def detect_damage(model: YOLO, frame) -> list:
    """
    Run damage detection on a single frame image.

    Parameters
    ----------
    model : YOLO model
    frame : numpy array (from cv2.VideoCapture)

    Returns
    -------
    list of raw YOLO results above confidence threshold
    """
    results = model.predict(
        source=frame,
        conf=CONFIDENCE_THRESHOLD,
        verbose=False,  # suppress per-frame console spam
    )
    return results


def process_frame(model: YOLO, extracted_frame) -> list:
    """
    Run detection on one ExtractedFrame.
    Returns list of Detection objects (empty if nothing found).
    """
    results = detect_damage(model, extracted_frame.image)
    detections = []

    for result in results:
        boxes = result.boxes

        if boxes is None or len(boxes) == 0:
            continue

        for box in boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            bbox = box.xyxy[0].tolist()

            damage_type = DAMAGE_LABELS.get(class_id, f"Unknown class {class_id}")

            detections.append(
                Detection(
                    damage_type=damage_type,
                    confidence=confidence,
                    bbox=bbox,
                    lat=extracted_frame.lat,
                    lon=extracted_frame.lon,
                    elevation=extracted_frame.elevation,
                    timestamp_utc=extracted_frame.timestamp_utc,
                    clip_filename=extracted_frame.clip_filename,
                    frame_number=extracted_frame.frame_number,
                    interpolated=extracted_frame.interpolated,
                )
            )

    return detections


def run_detection_on_clip(model: YOLO, extracted_frames: list) -> list:
    """
    Run detection across all frames from one clip.
    Returns all detections found.
    """
    all_detections = []
    frames_with_damage = 0

    for frame in extracted_frames:
        detections = process_frame(model, frame)

        if detections:
            frames_with_damage += 1
            all_detections.extend(detections)

    print(
        f"  🔍 Scanned {len(extracted_frames)} frames → "
        f"found {len(all_detections)} detections "
        f"in {frames_with_damage} frames"
    )

    return all_detections
