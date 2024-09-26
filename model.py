import torch

# Download the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Save the model locally
model.save("yolov5s.pt")

