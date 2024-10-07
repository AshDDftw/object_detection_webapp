from ultralytics import YOLO

# Load the YOLOv8 model for oriented bounding boxes (OBB)
model_path = "yolov8m-obb.pt"
model = YOLO(model_path)

# Global settings
confidence_threshold = 0.5  