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
import cv2


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


def interactive_crop(frame):
    def nothing(x):
        pass

    cv2.namedWindow("Adjust Crop")
    cv2.createTrackbar("y1", "Adjust Crop", 40, 100, nothing)
    cv2.createTrackbar("y2", "Adjust Crop", 85, 100, nothing)
    cv2.createTrackbar("x1", "Adjust Crop", 20, 100, nothing)
    cv2.createTrackbar("x2", "Adjust Crop", 80, 100, nothing)

    while True:
        h, w = frame.shape[:2]

        y1 = cv2.getTrackbarPos("y1", "Adjust Crop") / 100
        y2 = cv2.getTrackbarPos("y2", "Adjust Crop") / 100
        x1 = cv2.getTrackbarPos("x1", "Adjust Crop") / 100
        x2 = cv2.getTrackbarPos("x2", "Adjust Crop") / 100

        preview = frame.copy()

        cv2.rectangle(
            preview,
            (int(w * x1), int(h * y1)),
            (int(w * x2), int(h * y2)),
            (0, 255, 0),
            2,
        )

        cv2.imshow("Adjust Crop", preview)

        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC
            print("❌ Cancelled crop")
            break

        if key == 13:  # ENTER
            print("✅ Crop confirmed")
            break

    cv2.destroyAllWindows()
    return y1, y2, x1, x2


def process_frame(model: YOLO, extracted_frame, crop_values) -> list:
    """
    Run detection on one ExtractedFrame.
    Only considers detections in the road area (center-bottom of frame).
    """

    crop_y1, crop_y2, crop_x1, crop_x2 = crop_values
    height, width = extracted_frame.image.shape[:2]

    y1 = int(height * crop_y1)
    y2 = int(height * crop_y2)
    x1 = int(width * crop_x1)
    x2 = int(width * crop_x2)

    cropped = extracted_frame.image[y1:y2, x1:x2]

    results = detect_damage(model, cropped)

    detections = []

    height = extracted_frame.image.shape[0]
    width = extracted_frame.image.shape[1]

    for result in results:
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            continue

        for box in boxes:
            x1_box, y1_box, x2_box, y2_box = box.xyxy[0].tolist()

            x1_box += x1
            x2_box += x1
            y1_box += y1
            y2_box += y1

            center_x = (x1_box + x2_box) / 2
            center_y = (y1_box + y2_box) / 2

            box_width = x2_box - x1_box
            box_height = y2_box - y1_box

            if box_width > width * 0.25:
                continue
            if box_height > height * 0.25:
                continue

            confidence = float(box.conf[0])
            damage_type = DAMAGE_LABELS.get(int(box.cls[0]), "Unknown")

            detections.append(
                Detection(
                    damage_type=damage_type,
                    confidence=confidence,
                    bbox=[x1_box, y1_box, x2_box, y2_box],  # ✅ FIXED
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
    all_detections = []
    frames_with_damage = 0

    #  STEP 1 — get crop values ONCE
    first_frame = extracted_frames[0]
    crop_values = interactive_crop(first_frame.image)

    print(f"✅ Crop selected: {crop_values}")

    #  STEP 2 — loop frames
    for frame in extracted_frames:
        detections = process_frame(model, frame, crop_values)

        if detections:
            frames_with_damage += 1
            all_detections.extend(detections)

    print(
        f"  🔍 Scanned {len(extracted_frames)} frames → "
        f"found {len(all_detections)} detections "
        f"in {frames_with_damage} frames"
    )

    return all_detections
