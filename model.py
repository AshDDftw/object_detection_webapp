from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# Load the YOLOv8 model (YOLOv8 with oriented bounding boxes)
model = YOLO('yolov8m-obb.pt')

print(model.names.values())

# Function to run detection on an input image
def detect_objects(image_path, confidence_threshold=0.5):
    # Load image using OpenCV
    image = cv2.imread(image_path)

    # Check if the image was loaded successfully
    if image is None:
        print(f"Error loading image from {image_path}")
        return

    result = model.predict(image)
    boxes = result[0].obb

    # print(result)

    # Get the boxes, confidence scores, and class IDs
    # boxes = result[0].boxes.xyxy.cpu().numpy()  # Bounding box coordinates

    print(boxes.shape)


    

    # # Extract the annotated image with the detection results
    # annotated_image = result[0].plot()

    # # print(annotated_image)

    # # Visualize the image using matplotlib
    # plt.figure(figsize=(10, 10))
    # plt.imshow(annotated_image)
    # plt.axis('off')  # Hide axes
    # plt.show()

# Specify the path to your image
image_path = "for_testing_new_models"

# Run the object detection
detect_objects(image_path)
