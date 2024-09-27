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

# Load YOLOv5 model
model_path = "yolov5s.pt"  # Path to the saved YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True, skip_validation=True)

# Initialize Streamlit layout
st.set_page_config(
    page_title="Real-Time YOLOv5 Object Detection",
    page_icon="✅",
    layout="wide",
)

st.title("Object Detection Dashboard")

# Sidebar for options
st.sidebar.title("Settings")

# Option for class-wise selection
class_selection = st.sidebar.multiselect(
    "Select Classes to Detect",
    options=list(model.names.values()),  # Ensure it's a list
    default=list(model.names.values())   # Default to all classes
)

# Option for weather and sensor condition
weather_selection = st.sidebar.multiselect(
    "Select Weather and Sensor Conditions",
    options=["Rainy", "Cloudy", "Foggy"],
    default=[]  # No default selection
)

# Option to select static image/video upload or live detection
detection_mode = st.sidebar.radio(
    "Detection Mode",
    ("Live Detection", "Upload Image", "Upload Video")
)

# Confidence threshold slider
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)

# Create a grid layout for key metrics at the top
col1, col2, col3, col4 = st.columns(4)
total_objects = col1.metric("Total Objects", "0")
fps_display = col2.metric("FPS", "0")
resolution_display = col3.metric("Resolution", "0x0")
inference_speed_display = col4.metric("Inference Speed (ms)", "0")

# Initialize class count dictionary and dataframe for timestamp logging
class_count = {name: 0 for name in model.names.values()}
timestamp_data = {"time": [], "class": [], "count": []}

# Traffic data for graph per second
traffic_per_second = []
cumulative_traffic = []
start_time = datetime.now()

# Update the traffic graph and remove initial date on the x-axis
def update_traffic_graph(timestamp_data):
    df = pd.DataFrame(timestamp_data)
    
    df['time'] = pd.to_datetime(df['time'], format='%H:%M:%S')
    
    df_second = df.groupby(df['time'].dt.floor('S')).size().reset_index(name='count_per_second')
    df_second['cumulative_count'] = df_second['count_per_second'].cumsum()

    # Formatting to only show time without the date
    df_second['time'] = df_second['time'].dt.strftime('%H:%M:%S')

    traffic_graph = px.line(df_second, x='time', y='count_per_second', title="Traffic Per Second", template="plotly_dark")
    cumulative_graph = px.line(df_second, x='time', y='cumulative_count', title="Cumulative Traffic", template="plotly_dark")

    return traffic_graph, cumulative_graph

# Define the function to update the vehicle proportion pie chart
def update_vehicle_proportion_chart(class_count):
    vehicle_classes = ['car', 'truck', 'bus']  # You can adjust the vehicle classes
    vehicle_count = {cls: class_count[cls] for cls in vehicle_classes if cls in class_count and class_count[cls] > 0}
    
    if vehicle_count:
        vehicle_data = {"Vehicle Type": list(vehicle_count.keys()), "Count": list(vehicle_count.values())}
        df = pd.DataFrame(vehicle_data)
        fig = px.pie(df, names="Vehicle Type", values="Count", title="Proportion of Vehicle Types Detected",
                     template="plotly_dark", color_discrete_sequence=px.colors.sequential.RdBu)
    else:
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

# Define the function to update the histogram
def update_histogram(class_count):
    filtered_class_count = {k: v for k, v in class_count.items() if v > 0}
    
    if filtered_class_count:
        class_data = {
            "Classes": list(filtered_class_count.keys()),
            "Count": list(filtered_class_count.values())
        }
        df = pd.DataFrame(class_data)
        fig = px.bar(df, x="Classes", y="Count", title="Real-Time Object Detection Count",
                     color="Classes", template="plotly_dark")
    else:
        fig = px.bar(title="Waiting for Object Detection...", template="plotly_dark")
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
        fig = px.line(df, x="time", y="count", color="class", title="Object Detection Over Time",
                      template="plotly_dark")
    else:
        fig = px.line(title="No Data to Display Yet", template="plotly_dark")
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
        fig = px.pie(df, names="Classes", values="Area", title="Proportion of Frame Covered by Objects",
                     template="plotly_dark", color_discrete_sequence=px.colors.sequential.RdBu)
    else:
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

# Define the function to calculate inference speed
def calculate_inference_speed(start_time, end_time):
    return (end_time - start_time) * 1000  # Convert to milliseconds

# Define the function to calculate FPS
def calculate_fps(start_time, end_time):
    return 1 / (end_time - start_time) if (end_time - start_time) > 0 else 0

# Define the main detection function for processing frames
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
    current_time = datetime.now().strftime('%H:%M:%S')
    for *xyxy, conf, cls in results.xyxy[0]:
        if conf > confidence_threshold:
            class_name = model.names[int(cls)]
            if class_name in class_selection:
                class_count[class_name] += 1
                timestamp_data["time"].append(current_time)
                timestamp_data["class"].append(class_name)
                timestamp_data["count"].append(class_count[class_name])

    return results, class_count, frame_area, fps, inference_speed

# Initialize placeholders for the video and traffic data
with st.container():
    video_col, traffic_col = st.columns([2, 3])  # Adjust width to fit
    with video_col:
        frame_placeholder = st.empty()
    with traffic_col:
        traffic_graph_placeholder = st.empty()
        cumulative_graph_placeholder = st.empty()

with st.container():
    chart_col1, chart_col2, chart_col3 = st.columns(3)
    with chart_col1:
        histogram_placeholder = st.empty()
    with chart_col2:
        pie_chart_placeholder = st.empty()
    with chart_col3:
        vehicle_pie_chart_placeholder = st.empty()

# Initialize placeholder for the timestamp histogram
timestamp_histogram_placeholder = st.empty()

# Start main app logic based on the mode
if detection_mode == "Live Detection":
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    # Live detection loop
    while True:
        ret, frame = cap.read()
        if not ret:
            st.warning("Failed to read from webcam. Please check your webcam settings.")
            break

        results, current_class_count, frame_area, fps, inference_speed = process_frame(frame)

        total = sum(current_class_count.values())
        total_objects.metric("Total Objects", total)
        fps_display.metric("FPS", f"{fps:.2f}")
        resolution_display.metric("Resolution", f"{frame.shape[1]}x{frame.shape[0]}")
        inference_speed_display.metric("Inference Speed (ms)", f"{inference_speed:.2f}")

        for *xyxy, conf, cls in results.xyxy[0]:
            if conf > confidence_threshold and model.names[int(cls)] in class_selection:
                label = f'{model.names[int(cls)]} {conf:.2f}'
                cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])),
                              (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
                cv2.putText(frame, label, (int(xyxy[0]), int(xyxy[1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb_frame)
        frame_placeholder.image(image, caption="Webcam Feed", width=320)

        # Update and display the charts
        histogram = update_histogram(current_class_count)
        pie_chart = calculate_area_proportions(results, frame_area)
        vehicle_pie_chart = update_vehicle_proportion_chart(current_class_count)

        histogram_placeholder.plotly_chart(histogram, use_container_width=True)
        pie_chart_placeholder.plotly_chart(pie_chart, use_container_width=True)
        vehicle_pie_chart_placeholder.plotly_chart(vehicle_pie_chart, use_container_width=True)

        traffic_graph, cumulative_graph = update_traffic_graph(timestamp_data)
        traffic_graph_placeholder.plotly_chart(traffic_graph, use_container_width=True)
        cumulative_graph_placeholder.plotly_chart(cumulative_graph, use_container_width=True)

        line_graph = update_timestamp_line_graph(timestamp_data)
        timestamp_histogram_placeholder.plotly_chart(line_graph, use_container_width=True)

        time.sleep(0.1)

    cap.release()

elif detection_mode == "Upload Image":
    uploaded_image = st.sidebar.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        frame = np.array(image)
        
        results, current_class_count, frame_area, fps, inference_speed = process_frame(frame)
        
        total = sum(current_class_count.values())
        total_objects.metric("Total Objects", total)
        fps_display.metric("FPS", f"{fps:.2f}")
        resolution_display.metric("Resolution", f"{frame.shape[1]}x{frame.shape[0]}")
        inference_speed_display.metric("Inference Speed (ms)", f"{inference_speed:.2f}")

        for *xyxy, conf, cls in results.xyxy[0]:
            if conf > confidence_threshold and model.names[int(cls)] in class_selection:
                label = f'{model.names[int(cls)]} {conf:.2f}'
                cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])),
                              (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
                cv2.putText(frame, label, (int(xyxy[0]), int(xyxy[1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_with_detections = Image.fromarray(rgb_frame)
        st.image(image_with_detections, caption="Uploaded Image", use_column_width=True)

        histogram = update_histogram(current_class_count)
        pie_chart = calculate_area_proportions(results, frame_area)
        vehicle_pie_chart = update_vehicle_proportion_chart(current_class_count)

        histogram_placeholder.plotly_chart(histogram, use_container_width=True)
        pie_chart_placeholder.plotly_chart(pie_chart, use_container_width=True)
        vehicle_pie_chart_placeholder.plotly_chart(vehicle_pie_chart, use_container_width=True)

        traffic_graph, cumulative_graph = update_traffic_graph(timestamp_data)
        traffic_graph_placeholder.plotly_chart(traffic_graph, use_container_width=True)
        cumulative_graph_placeholder.plotly_chart(cumulative_graph, use_container_width=True)

        line_graph = update_timestamp_line_graph(timestamp_data)
        timestamp_histogram_placeholder.plotly_chart(line_graph, use_container_width=True)

elif detection_mode == "Upload Video":
    uploaded_video = st.sidebar.file_uploader("Upload a Video", type=["mp4", "avi", "mov"])
    
    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        
        cap = cv2.VideoCapture(tfile.name)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.success("Video processing completed.")
                break
            
            results, current_class_count, frame_area, fps, inference_speed = process_frame(frame)
            
            total = sum(current_class_count.values())
            total_objects.metric("Total Objects", total)
            fps_display.metric("FPS", f"{fps:.2f}")
            resolution_display.metric("Resolution", f"{frame.shape[1]}x{frame.shape[0]}")
            inference_speed_display.metric("Inference Speed (ms)", f"{inference_speed:.2f}")

            for *xyxy, conf, cls in results.xyxy[0]:
                if conf > confidence_threshold and model.names[int(cls)] in class_selection:
                    label = f'{model.names[int(cls)]} {conf:.2f}'
                    cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])),
                                  (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
                    cv2.putText(frame, label, (int(xyxy[0]), int(xyxy[1]) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb_frame)
            frame_placeholder.image(image, caption="Video Frame", width=320)

            histogram = update_histogram(current_class_count)
            pie_chart = calculate_area_proportions(results, frame_area)
            vehicle_pie_chart = update_vehicle_proportion_chart(current_class_count)

            histogram_placeholder.plotly_chart(histogram, use_container_width=True)
            pie_chart_placeholder.plotly_chart(pie_chart, use_container_width=True)
            vehicle_pie_chart_placeholder.plotly_chart(vehicle_pie_chart, use_container_width=True)

            traffic_graph, cumulative_graph = update_traffic_graph(timestamp_data)
            traffic_graph_placeholder.plotly_chart(traffic_graph, use_container_width=True)
            cumulative_graph_placeholder.plotly_chart(cumulative_graph, use_container_width=True)

            line_graph = update_timestamp_line_graph(timestamp_data)
            timestamp_histogram_placeholder.plotly_chart(line_graph, use_container_width=True)

            time.sleep(0.03)

        cap.release()
