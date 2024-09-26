import time
import torch
import numpy as np
import cv2
import streamlit as st
import plotly.express as px
from PIL import Image
import pandas as pd
from datetime import datetime
import tempfile

model_path = "yolov5s.pt"  # Path to the saved YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)

# Initialize Streamlit layout
st.set_page_config(
    page_title="Real-Time YOLOv5 Object Detection",
    page_icon="✅",
    layout="wide",
)

st.title("Object Detection App")

# Sidebar for options
st.sidebar.title("Settings")

# Option for class-wise selection
class_selection = st.sidebar.multiselect(
    "Select Classes to Detect",
    options=model.names.values(), 
      # Default to all classes
)

# Option to select static image/video upload or live detection
detection_mode = st.sidebar.radio(
    "Detection Mode", 
    ("Live Detection", "Upload Image", "Upload Video")
)

# Confidence threshold slider
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)

# Create two-column layout
col1, col2 = st.columns([1, 2])

# Initialize class count dictionary and dataframe for timestamp logging
class_count = {name: 0 for name in model.names}
timestamp_data = {"time": [], "class": [], "count": []}

# Define the function to update the histogram
def update_histogram(class_count):
    # Filter out classes that have zero count
    filtered_class_count = {k: v for k, v in class_count.items() if v > 0}
    
    if filtered_class_count:
        class_data = {
            "Classes": list(filtered_class_count.keys()),
            "Count": list(filtered_class_count.values())
        }
        df = pd.DataFrame(class_data)
        fig = px.bar(df, x="Classes", y="Count", title="Real-Time Object Detection Count")
    else:
        fig = px.bar(title="Waiting for Object Detection...")
        fig.update_layout(xaxis={'visible': False}, yaxis={'visible': False}, annotations=[{
            'text': 'No Objects Detected',
            'xref': 'paper',
            'yref': 'paper',
            'showarrow': False,
            'font': {'size': 28}
        }])
    return fig

# Define the function to update the timestamp line graph
def update_timestamp_line_graph(timestamp_data):
    df = pd.DataFrame(timestamp_data)
    if not df.empty:
        fig = px.line(df, x="time", y="count", color="class", title="Object Detection Over Time")
    else:
        fig = px.line(title="No Data to Display Yet")
        fig.update_layout(xaxis={'visible': False}, yaxis={'visible': False}, annotations=[{
            'text': 'No Objects Detected Yet',
            'xref': 'paper',
            'yref': 'paper',
            'showarrow': False,
            'font': {'size': 28}
        }])
    return fig

# Define the function to calculate area proportions and plot a pie chart
def calculate_area_proportions(results, frame_area):
    area_dict = {name: 0 for name in model.names.values()}
    
    for *xyxy, conf, cls in results.xyxy[0]:
        if conf > confidence_threshold:  # Use confidence threshold
            class_name = model.names[int(cls)]
            if class_name in class_selection:
                # Calculate bounding box area
                width = xyxy[2] - xyxy[0]
                height = xyxy[3] - xyxy[1]
                area = width * height
                area_dict[class_name] += area

    filtered_area_dict = {k: v for k, v in area_dict.items() if v > 0}
    
    if filtered_area_dict:
        area_data = {
            "Classes": list(filtered_area_dict.keys()),
            "Area": [area / frame_area * 100 for area in filtered_area_dict.values()]
        }
        df = pd.DataFrame(area_data)
        fig = px.pie(df, names="Classes", values="Area", title="Proportion of Frame Covered by Objects")
    else:
        fig = px.pie(title="No Objects Detected")
        fig.update_traces(marker=dict(colors=['grey']))
        fig.update_layout(annotations=[{
            'text': 'No Objects Detected',
            'xref': 'paper',
            'yref': 'paper',
            'showarrow': False,
            'font': {'size': 28}
        }])
    
    return fig

# Main detection function for processing frames
def process_frame(frame):
    frame_area = frame.shape[0] * frame.shape[1]  # Calculate frame area
    
    results = model(frame)
    
    # Reset class counts
    class_count = {name: 0 for name in model.names.values()}

    # Update class counts and timestamp log
    current_time = datetime.now().strftime('%H:%M:%S')
    for *xyxy, conf, cls in results.xyxy[0]:
        if conf > confidence_threshold:
            class_name = model.names[int(cls)]
            if class_name in class_selection:
                class_count[class_name] += 1
                timestamp_data["time"].append(current_time)
                timestamp_data["class"].append(class_name)
                timestamp_data["count"].append(class_count[class_name])

    return results, class_count, frame_area

# Start main app logic based on the mode
if detection_mode == "Live Detection":
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    with col1:
        frame_placeholder = st.empty()
    with col2:
        histogram_placeholder = st.empty()
        line_graph_placeholder = st.empty()
        area_chart_placeholder = st.empty()

    # Live detection loop
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process the frame
        results, class_count, frame_area = process_frame(frame)

        # Display detection results
        for *xyxy, conf, cls in results.xyxy[0]:
            if conf > confidence_threshold and model.names[int(cls)] in class_selection:
                label = f'{model.names[int(cls)]} {conf:.2f}'
                cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
                cv2.putText(frame, label, (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb_frame)
        frame_placeholder.image(image, caption="Webcam Feed", width=320)

        # Update and display the charts
        histogram = update_histogram(class_count)
        histogram_placeholder.write(histogram)
        line_graph = update_timestamp_line_graph(timestamp_data)
        line_graph_placeholder.write(line_graph)
        area_chart = calculate_area_proportions(results, frame_area)
        area_chart_placeholder.write(area_chart)

        time.sleep(0.5)

    cap.release()

elif detection_mode == "Upload Image":
    uploaded_image = st.sidebar.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        frame = np.array(image)
        
        results, class_count, frame_area = process_frame(frame)
        
        # Display the image and results
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Display charts
        histogram = update_histogram(class_count)
        col2.write(histogram)
        line_graph = update_timestamp_line_graph(timestamp_data)
        col2.write(line_graph)
        area_chart = calculate_area_proportions(results, frame_area)
        col2.write(area_chart)

elif detection_mode == "Upload Video":
    uploaded_video = st.sidebar.file_uploader("Upload a Video", type=["mp4", "avi", "mov"])
    
    if uploaded_video is not None:
        # Create a temporary file to store the video
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        
        cap = cv2.VideoCapture(tfile.name)
        
        with col1:
            frame_placeholder = st.empty()
        with col2:
            histogram_placeholder = st.empty()
            line_graph_placeholder = st.empty()
            area_chart_placeholder = st.empty()

        # Video processing loop
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            results, class_count, frame_area = process_frame(frame)
            
            # Display detection results on the frame
            for *xyxy, conf, cls in results.xyxy[0]:
                if conf > confidence_threshold and model.names[int(cls)] in class_selection:
                    label = f'{model.names[int(cls)]} {conf:.2f}'
                    cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
                    cv2.putText(frame, label, (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb_frame)
            frame_placeholder.image(image, caption="Video Frame", width=320)

            # Update and display charts
            histogram = update_histogram(class_count)
            histogram_placeholder.write(histogram)
            line_graph = update_timestamp_line_graph(timestamp_data)
            line_graph_placeholder.write(line_graph)
            area_chart = calculate_area_proportions(results, frame_area)
            area_chart_placeholder.write(area_chart)

        cap.release()
