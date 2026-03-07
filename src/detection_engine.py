"""
detection_engine.py
-------------------
Runs YOLOv8 pothole detection on extracted frames.
Uses a fine-tuned YOLOv8 model trained on pothole detection.

Note: Initially designed for RDD2022 multi-class road damage detection
(cracks, potholes). Narrowed to pothole detection only after testing
showed better precision with a dedicated pothole model in Toronto
winter conditions.
"""

from ultralytics import YOLO
from dataclasses import dataclass
import urllib.request
import os


CONFIDENCE_THRESHOLD = 0.45

# ignore detections below 35%
# too low = too many false positives
# too high = miss real damage
MODEL_PATH = "models/rdd2022_yolov8.pt"


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

    # We fine-tune it toward road damage via the RDD2022 weights
    model = YOLO("yolov8n.pt")

    print(f"✅ Base model ready")
    print("   Note: For best results on Toronto roads, the model")
    print("   uses YOLOv8 trained on RDD2022 international dataset")


def load_model() -> YOLO:
    """
    Load the Pothole-Finetuned-YoloV8 model.
    """
    model_path = "models/Yolov8-fintuned-on-potholes.pt"

    if not os.path.exists(model_path):
        print("📥 Downloading model...")
        from huggingface_hub import login, hf_hub_download

        token = os.getenv("HF_TOKEN")
        if token is None:
            raise ValueError(
                "HF_TOKEN environment variable not set. Add it to your .env file."
            )
        login(token=token)
        hf_hub_download(
            repo_id="cazzz307/Pothole-Finetuned-YoloV8",
            filename="Yolov8-fintuned-on-potholes.pt",
            local_dir="models",
        )
        print("✅ Downloaded")

    model = YOLO(model_path)
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
    Only considers detections in the road area (center-bottom of frame).
    """
    results = detect_damage(model, extracted_frame.image)
    detections = []

    height = extracted_frame.image.shape[0]
    width = extracted_frame.image.shape[1]

    # Road zone — where actual road surface in front of car appears
    # Ignore top 40% (sky, buildings), ignore side 20% (sidewalks)
    road_x_min = width * 0.20  # ignore left 20%
    road_x_max = width * 0.80  # ignore right 20%
    road_y_min = height * 0.40  # ignore top 40%
    # Skip unrealistically large detections
    # A real pothole is never more than 25% of frame width

    for result in results:
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            continue

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            # Calculate center of detection box
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            # Skip if detection center is outside road zone
            if center_x < road_x_min or center_x > road_x_max:
                continue  # too far left or right
            if center_y < road_y_min:
                continue  # too high up (sky/buildings)
            box_width = x2 - x1
            box_height = y2 - y1

            if box_width > width * 0.25:
                continue  # too wide to be a pothole
            if box_height > height * 0.25:
                continue  # too tall to be a pothole
            confidence = float(box.conf[0])
            damage_type = DAMAGE_LABELS.get(int(box.cls[0]), "Unknown")

            detections.append(
                Detection(
                    damage_type=damage_type,
                    confidence=confidence,
                    bbox=[x1, y1, x2, y2],
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
