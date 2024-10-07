from ultralytics import YOLO

# Download the YOLOv8 model (YOLOv8s in this case)
model = YOLO('yolov8m-obb.pt') 

model.export() 
