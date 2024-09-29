import torch
import pandas as pd
import cv2
import time
from datetime import datetime
import plotly.express as px
import numpy as np
from PIL import Image

# Load YOLOv5 model
model_path = "yolov5s.pt"  # Path to the saved YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True, skip_validation=True)

# Confidence threshold slider
confidence_threshold = 0.5

# Initialize class count dictionary and dataframe for timestamp logging
class_count = {name: 0 for name in model.names.values()}
timestamp_data = {"time": [], "class": [], "count": []}

import plotly.express as px
import pandas as pd

def update_vehicle_proportion_chart(class_count):
    """
    Generates a pie chart representing the proportion of detected vehicle types.

    Args:
        class_count: A dictionary containing the count of detected objects by class.
    
    Returns:
        fig: Plotly pie chart figure.
    """
    # Define the vehicle classes (adjust if necessary)
    vehicle_classes = ['car', 'truck', 'bus']  # Adjust to match your specific vehicle classes

    # Filter out the detected vehicle classes from the class_count dictionary
    vehicle_count = {cls: class_count[cls] for cls in vehicle_classes if cls in class_count and class_count[cls] > 0}
    
    if vehicle_count:
        # Prepare data for the pie chart
        vehicle_data = {"Vehicle Type": list(vehicle_count.keys()), "Count": list(vehicle_count.values())}
        df = pd.DataFrame(vehicle_data)
        # Generate the pie chart using Plotly
        fig = px.pie(df, names="Vehicle Type", values="Count", title="Proportion of Vehicle Types Detected",
                     template="plotly_dark", color_discrete_sequence=px.colors.sequential.RdBu)
    else:
        # If no vehicles are detected, return an empty pie chart
        fig = px.pie(title="No Vehicles Detected", template="plotly_dark")
        fig.update_traces(marker=dict(colors=['grey']))
        fig.update_layout(annotations=[{
            'text': 'No Vehicles Detected',
            'xref': 'paper',
            'yref': 'paper',
            'showarrow': False,
            'font': {'size': 28}
        }])
    
    return fig

def calculate_area_proportions(results, frame_area):
    """
    Calculate the proportions of the frame covered by detected objects.
    
    Args:
        results: YOLOv5 detection results
        frame_area: The total area of the frame
    
    Returns:
        fig: Plotly pie chart figure
    """
    # Initialize a dictionary to store the area covered by each class
    area_dict = {name: 0 for name in model.names.values()}
    
    for *xyxy, conf, cls in results.xyxy[0]:
        if conf > confidence_threshold:  # Use confidence threshold
            class_name = model.names[int(cls)]
            # Calculate bounding box area
            width = xyxy[2] - xyxy[0]
            height = xyxy[3] - xyxy[1]
            area = width * height
            area_dict[class_name] += area

    # Filter out classes with no detected areas
    filtered_area_dict = {k: v for k, v in area_dict.items() if v > 0}
    
    # If there are any detections, generate the pie chart
    if filtered_area_dict:
        area_data = {
            "Classes": list(filtered_area_dict.keys()),
            "Area": [area / frame_area * 100 for area in filtered_area_dict.values()]
        }
        df = pd.DataFrame(area_data)
        fig = px.pie(df, names="Classes", values="Area", title="Proportion of Frame Covered by Objects",
                     template="plotly_dark", color_discrete_sequence=px.colors.sequential.RdBu)
    else:
        # No objects detected
        fig = px.pie(title="No Objects Detected", template="plotly_dark")
        fig.update_traces(marker=dict(colors=['grey']))
        fig.update_layout(annotations=[{
            'text': 'No Objects Detected',
            'xref': 'paper',
            'yref': 'paper',
            'showarrow': False,
            'font': {'size': 28}
        }])
    
    return fig


# Define process_frame function and others
def process_frame(frame):
    start_time = time.time()

    results = model(frame)
    end_time = time.time()

    inference_speed = calculate_inference_speed(start_time, end_time)
    fps = calculate_fps(start_time, end_time)

    frame_area = frame.shape[0] * frame.shape[1]  # Calculate frame area

    # Reset class counts
    class_count = {name: 0 for name in model.names.values()}

    # Update class counts and timestamp log
    current_time = datetime.now().strftime("%H:%M:%S")
    for *xyxy, conf, cls in results.xyxy[0]:
        if conf > confidence_threshold:
            class_name = model.names[int(cls)]
            class_count[class_name] += 1
            timestamp_data["time"].append(current_time)
            timestamp_data["class"].append(class_name)
            timestamp_data["count"].append(class_count[class_name])

    return results, class_count, frame_area, fps, inference_speed


def calculate_inference_speed(start_time, end_time):
    return (end_time - start_time) * 1000


def calculate_fps(start_time, end_time):
    return 1 / (end_time - start_time) if (end_time - start_time) > 0 else 0


# Other functions like update_histogram, update_traffic_graph, etc.
def update_histogram(class_count):
    filtered_class_count = {k: v for k, v in class_count.items() if v > 0}

    if filtered_class_count:
        class_data = {"Classes": list(filtered_class_count.keys()), "Count": list(filtered_class_count.values())}
        df = pd.DataFrame(class_data)
        fig = px.bar(df, x="Classes", y="Count", title="Real-Time Object Detection Count", color="Classes", template="plotly_dark")
    else:
        fig = px.bar(title="Waiting for Object Detection...", template="plotly_dark")
        fig.update_layout(xaxis={"visible": False}, yaxis={"visible": False})

    return fig


def update_traffic_graph(timestamp_data):
    df = pd.DataFrame(timestamp_data)
    df["time"] = pd.to_datetime(df["time"], format="%H:%M:%S")
    df_second = df.groupby(df["time"].dt.floor("S")).size().reset_index(name="count_per_second")
    df_second["cumulative_count"] = df_second["count_per_second"].cumsum()
    df_second["time"] = df_second["time"].dt.strftime("%H:%M:%S")
    traffic_graph = px.line(df_second, x="time", y="count_per_second", title="Traffic Per Second", template="plotly_dark")
    cumulative_graph = px.line(df_second, x="time", y="cumulative_count", title="Cumulative Traffic", template="plotly_dark")

    return traffic_graph, cumulative_graph

# Define other utility functions similarly...
