import time
import cv2
import torch
import random
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
from streamlit_webrtc import VideoProcessorBase
import threading
import av
import streamlit as st
import config



# Global variables to track traffic data
timestamp_data = {"time": [], "class": [], "count": []}
color_map = {}

# Function to process a single frame (image/video)
# Function to process a single frame (image/video)
def process_frame(frame, model, confidence_threshold, class_selection):
    start_time = time.time()

    # print('hereeee')

    # Object detection on the frame
    results = model(frame)
    end_time = time.time()

    # Calculate metrics
    inference_speed = calculate_inference_speed(start_time, end_time)
    fps = calculate_fps(start_time, end_time)
    frame_area = frame.shape[0] * frame.shape[1]

    class_count = {name: 0 for name in model.names.values()}
    timestamp_data = {"time": [], "class": [], "count": []}

    # Update class counts and timestamp log
    current_time = datetime.now().strftime('%H:%M:%S')
    objects_detected = False  # Track if any objects are detected in this frame

    for *xyxy, conf, cls in results.xyxy[0]:
        if conf > confidence_threshold:
            class_name = model.names[int(cls)]
            if class_name in class_selection:
                class_count[class_name] += 1
                timestamp_data["time"].append(current_time)
                timestamp_data["class"].append(class_name)
                timestamp_data["count"].append(class_count[class_name])
                objects_detected = True

    # If no objects are detected, log zero count for this time
    if not objects_detected:
        timestamp_data["time"].append(current_time)
        timestamp_data["class"].append("None")  # Optional, you can add a placeholder class
        timestamp_data["count"].append(0)

    # print(timestamp_data)

    return results, class_count, frame_area, fps, inference_speed, timestamp_data


class YOLOv5VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.confidence_threshold = 0.5  # Set a default confidence threshold
        self.class_selection = []  # Selected classes
        self.lock = threading.Lock()  # Ensure thread-safe access
        self.results = None
        self.current_class_count = {}
        self.fps = 0
        self.inference_speed = 0
        self.frame_area = 0
        self.model = config.model  # Load YOLOv5 model

    def update_params(self, confidence_threshold, class_selection):
        with self.lock:
            self.confidence_threshold = confidence_threshold
            self.class_selection = class_selection

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")  # Get the video frame as a NumPy array

        # Debugging: Check if the frame is captured
        print("Captured frame shape:", img.shape)

        # Process frame and collect metrics
        try:
            results, current_class_count, frame_area, fps, inference_speed, _ = process_frame(
                img, self.model, self.confidence_threshold, self.class_selection
            )
        except Exception as e:
            print("Error in frame processing:", str(e))
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        # Update class state safely with a lock
        with self.lock:
            self.results = results
            self.current_class_count = current_class_count
            self.frame_area = frame_area
            self.fps = fps
            self.inference_speed = inference_speed

        # Draw bounding boxes and labels on the frame
        for *xyxy, conf, cls in results.xyxy[0]:
            if conf > self.confidence_threshold:  # Apply the confidence threshold
                class_name = self.model.names[int(cls)]
                if class_name in self.class_selection:
                    label = f'{class_name} {conf:.2f}'
                    cv2.rectangle(img, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
                    cv2.putText(img, label, (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Debugging: Ensure the image is properly processed
        print("Processed frame ready for rendering")

        # Return the processed frame
        return av.VideoFrame.from_ndarray(img, format="bgr24")


# Function to update the metric blocks
def update_metric_blocks(total_objects, fps_value, resolution_width, resolution_height, inference_speed_value):
    st.markdown(f"""
        <div style="background-color: #f94144; padding: 10px; border-radius: 10px;margin:10px color: white; text-align: center;">
            <h3>Total Objects</h3>
            <h2>{total_objects}</h2>
        </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
        <div style="background-color: #f3722c; padding: 10px; border-radius: 10px; color: white; text-align: center;">
            <h3>FPS</h3>
            <h2>{fps_value:.2f}</h2>
        </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
        <div style="background-color: #f9c74f; padding: 10px; border-radius: 10px; color: black; text-align: center;">
            <h3>Resolution</h3>
            <h2>{resolution_width}x{resolution_height}</h2>
        </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
        <div style="background-color: #43aa8b; padding: 10px; border-radius: 10px;margin-bottom:10px color: white; text-align: center;">
            <h3>Inference(ms)</h3>
            <h2>{inference_speed_value:.2f}</h2>
        </div>
    """, unsafe_allow_html=True)

# Function to calculate inference speed
def calculate_inference_speed(start_time, end_time):
    return (end_time - start_time) * 1000  # Convert to milliseconds

# Function to calculate FPS
def calculate_fps(start_time, end_time):
    return 1 / (end_time - start_time) if (end_time - start_time) > 0 else 0

# Function to update the histogram
def update_histogram(class_count):
    filtered_class_count = {k: v for k, v in class_count.items() if v > 0}

    if filtered_class_count:
        class_data = {
            "Classes": list(filtered_class_count.keys()),
            "Count": list(filtered_class_count.values())
        }
        df = pd.DataFrame(class_data)
        fig = px.bar(df, x="Classes", y="Count", title="Object Count",
                     color="Classes", template="plotly_dark", height=250)
    else:
        fig = px.bar(title="Waiting for Object Detection...", template="plotly_dark", height=250)
        fig.update_layout(xaxis={'visible': False}, yaxis={'visible': False}, annotations=[{
            'text': 'No Objects Detected',
            'xref': 'paper',
            'yref': 'paper',
            'showarrow': False,
            'font': {'size': 16}
        }])
    return fig

# Function to update the vehicle proportion pie chart
def update_vehicle_proportion_chart(class_count):
    vehicle_classes = ['car', 'truck', 'bus', 'motorcycle']  # Adjust vehicle classes as needed
    vehicle_count = {cls: class_count[cls] for cls in vehicle_classes if cls in class_count and class_count[cls] > 0}

    if vehicle_count:
        vehicle_data = {"Vehicle Type": list(vehicle_count.keys()), "Count": list(vehicle_count.values())}
        df = pd.DataFrame(vehicle_data)
        fig = px.pie(df, names="Vehicle Type", values="Count", title="Vehicle Proportion",
                     template="plotly_dark", color_discrete_sequence=px.colors.sequential.RdBu, height=325)
    else:
        fig = px.pie(title="No Vehicles Detected", template="plotly_dark", height=325)
        fig.update_traces(marker=dict(colors=['grey']))
        fig.update_layout(annotations=[{
            'text': 'No Vehicles Detected',
            'xref': 'paper',
            'yref': 'paper',
            'showarrow': False,
            'font': {'size': 16}
        }])
    return fig

# Function to calculate area proportions and plot a pie chart
def calculate_area_proportions(results, frame_area, class_selection):
    # Initialize area_dict for only the selected classes
    area_dict = {name: 0 for name in class_selection}

    for *xyxy, conf, cls in results.xyxy[0]:
        if conf > 0.5:  # Use confidence threshold
            class_name = config.model.names[int(cls)]
            if class_name in class_selection:  # Only calculate area for selected classes
                # Calculate bounding box area
                width = xyxy[2] - xyxy[0]
                height = xyxy[3] - xyxy[1]
                area = width * height
                area_dict[class_name] += area

    # Filter out classes with 0 area
    filtered_area_dict = {k: v for k, v in area_dict.items() if v > 0}

    # If any areas were calculated, create the pie chart
    if filtered_area_dict:
        area_data = {
            "Classes": list(filtered_area_dict.keys()),
            "Area": [area / frame_area * 100 for area in filtered_area_dict.values()]
        }
        df = pd.DataFrame(area_data)
        fig = px.pie(df, names="Classes", values="Area", title="Area Proportion",
                     template="plotly_dark", color_discrete_sequence=px.colors.sequential.RdBu, height=280)
    else:
        # If no objects were detected for the selected classes, show an empty pie chart
        fig = px.pie(title="No Objects Detected", template="plotly_dark", height=290)
        fig.update_traces(marker=dict(colors=['grey']))
        fig.update_layout(annotations=[{
            'text': 'No Objects Detected',
            'xref': 'paper',
            'yref': 'paper',
            'showarrow': False,
            'font': {'size': 16}
        }])

    return fig


def update_traffic_graph(timestamp_data):
    df = pd.DataFrame(timestamp_data)

    # Convert 'time' to datetime for grouping
    df['time'] = pd.to_datetime(df['time'], format='%H:%M:%S')

    # Group by second and sum counts (handles missing data with zero counts)
    df_second = df.groupby(df['time'].dt.floor('S')).sum().reset_index()

    # Ensure all time intervals have data, even with zero counts
    time_range = pd.date_range(df['time'].min(), df['time'].max(), freq='S')
    df_second = df.groupby(df['time'].dt.floor('S')).size().reset_index(name='count_per_second')
    df_second.columns = ['time', 'count_per_second']

    # Calculate cumulative count
    df_second['cumulative_count'] = df_second['count_per_second'].cumsum()

    # Formatting to only show time without the date
    df_second['time'] = df_second['time'].dt.strftime('%H:%M:%S')

    # Generate the traffic graph and cumulative graph
    traffic_graph = px.line(df_second, x='time', y='count_per_second', title="Traffic Per Second", template="plotly_dark", height=250)
    cumulative_graph = px.line(df_second, x='time', y='cumulative_count', title="Cumulative Traffic", template="plotly_dark", height=325)

    return traffic_graph, cumulative_graph

# Function to generate a random color for each class
def get_random_color():
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

# Layout for three aligned plots at the bottom
def update_bottom_plots(histogram, pie_chart, vehicle_pie_chart, traffic_graph, cumulative_graph):
    col1, col2, col3 = st.columns(3)

    with col1:
        histogram.plotly_chart(use_container_width=True)
    with col2:
        pie_chart.plotly_chart(use_container_width=True)
    with col3:
        vehicle_pie_chart.plotly_chart(use_container_width=True)

    col4, col5 = st.columns(2)
    with col4:
        traffic_graph.plotly_chart(use_container_width=True)
    with col5:
        cumulative_graph.plotly_chart(use_container_width=True)