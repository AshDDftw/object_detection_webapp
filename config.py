# config.py

import torch

# Load the YOLOv5 model once
model_path = "yolov5s.pt"
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True, skip_validation=True)

# You can also define other global settings here, like confidence_threshold or class_selection
confidence_threshold = 0.5  # Default confidence threshold
