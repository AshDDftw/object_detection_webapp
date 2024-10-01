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
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av
import threading

# Load YOLOv5 model
model_path = "yolov5s.pt"  # Path to the saved YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True, skip_validation=True)

# Initialize Streamlit layout
st.set_page_config(
    page_title="Real-Time YOLOv5 Object Detection",
    page_icon="✅",
    layout="wide",
)

st.markdown("""
    <style>
    .css-18e3th9 { padding-top: 0.5rem; padding-bottom: 0.5rem; }
    .css-1d391kg { padding-top: 0.5rem; padding-bottom: 0.5rem; }
    .stApp { background-color: #6b38b5; padding-left: 0px; padding-right: 0px;} /* Light background */

    /* Style the title */
    .title {
        font-family: 'Roboto', sans-serif;
        color: #FBE9D0;
        font-size: 3rem;
        font-weight: 700;
        text-align: center;
        margin-top: 20px;
        margin-bottom: 20px;
        background-color:#FBE9D0:
    }

    /* Make the content edge-to-edge */
    .main {
        padding-left: 0px;
        padding-right: 0px;
    }

    /* Professional colors for blocks */
    .block { 
        background-color: #2E3B4E; 
        padding: 10px; 
        border-radius: 10px; 
        color: white; 
        text-align: center; 
        margin-bottom: 10px;
    }
    .block-orange { 
        background-color: #F39C12; 
        padding: 10px; 
        border-radius: 10px; 
        color: white; 
        text-align: center; 
        margin-bottom: 10px;
    }
    .block-yellow { 
        background-color: #F1C40F; 
        padding: 10px; 
        border-radius: 10px; 
        color: black; 
        text-align: center; 
        margin-bottom: 10px;
    }
    .block-green { 
        background-color: #27AE60; 
        padding: 10px; 
        border-radius: 10px; 
        color: white; 
        text-align: center; 
        margin-bottom: 10px;
    }

    /* Remove padding and margin at the bottom to prevent large gap */
    .main .block-container {
        padding-bottom: 0px !important;
        margin-bottom: 0px !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Add title with new styling
st.markdown('<h1 class="title">Object Detection Dashboard</h1>', unsafe_allow_html=True)
# Sidebar for options
st.sidebar.title("Settings")

# Option for class-wise selection (no default selection)
class_selection = st.sidebar.multiselect(
    "Select Classes to Detect",
    options=list(model.names.values()),  # List all available class names in the multiselect options
    default=None  # No default classes pre-selected
)

# Ensure all classes are detected if no specific classes are selected
if not class_selection or len(class_selection) == 0:
    class_selection = list(model.names.values())  # Detect all classes by default

# Option to select static image/video upload or live detection
detection_mode = st.sidebar.radio(
    "Detection Mode",
    ("Live Detection", "Upload Image", "Upload Video")
)

# Add weather dropdown to the sidebar
weather_conditions = st.sidebar.selectbox(
    "Select Weather Condition",
    ("Normal", "Rainy", "Cloudy", "Foggy", "Other")
)

# Confidence threshold slider
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)

# Create placeholders for the metrics to update later

# Variables to hold actual values
total_objects = 0
fps_display = 0.0
resolution_width = 0
resolution_height = 0
inference_speed_display = 0.0

# Create placeholders for the metrics to update later
col1, col2, col3, col4 = st.columns([1, 1, 1, 1])

with col1:
    total_objects_placeholder = st.empty()

with col2:
    fps_placeholder = st.empty()

with col3:
    resolution_placeholder = st.empty()

with col4:
    inference_speed_placeholder = st.empty()

# Function to update the placeholders with the real values dynamically
def update_metric_blocks(total_objects, fps_value, resolution_width, resolution_height, inference_speed_value):
    total_objects_placeholder.markdown(f"""
        <div style="background-color: #f94144; padding: 10px; border-radius: 10px;margin:10px color: white; text-align: center;">
            <h3>Total Objects</h3>
            <h2>{total_objects}</h2>
        </div>
    """, unsafe_allow_html=True)

    fps_placeholder.markdown(f"""
        <div style="background-color: #f3722c; padding: 10px; border-radius: 10px; color: white; text-align: center;">
            <h3>FPS</h3>
            <h2>{fps_value:.2f}</h2>
        </div>
    """, unsafe_allow_html=True)

    resolution_placeholder.markdown(f"""
        <div style="background-color: #f9c74f; padding: 10px; border-radius: 10px; color: black; text-align: center;">
            <h3>Resolution</h3>
            <h2>{resolution_width}x{resolution_height}</h2>
        </div>
    """, unsafe_allow_html=True)

    inference_speed_placeholder.markdown(f"""
        <div style="background-color: #43aa8b; padding: 10px; border-radius: 10px;margin-bottom:10px color: white; text-align: center;">
            <h3>Inference(ms)</h3>
            <h2>{inference_speed_value:.2f}</h2>
        </div>
    """, unsafe_allow_html=True)

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


# Add the rest of the logic here to process frames, display results, and align the boxes as needed
# Ensure the final section uses the update_bottom_plots function to align your elements properly


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

    traffic_graph = px.line(df_second, x='time', y='count_per_second', title="Traffic Per Second", template="plotly_dark", height=250)
    cumulative_graph = px.line(df_second, x='time', y='cumulative_count', title="Cumulative Traffic", template="plotly_dark", height=325)

    return traffic_graph, cumulative_graph

# Define the function to update the vehicle proportion pie chart
def update_vehicle_proportion_chart(class_count):
    vehicle_classes = ['car', 'truck', 'bus','motorcyclew']  # You can adjust the vehicle classes
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

# Define the function to update the histogram
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
        fig = px.pie(df, names="Classes", values="Area", title="Area Proportion",
                     template="plotly_dark", color_discrete_sequence=px.colors.sequential.RdBu, height=280)
    else:
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

# Define the function to calculate inference speed
def calculate_inference_speed(start_time, end_time):
    return (end_time - start_time) * 1000  # Convert to milliseconds

# Define the function to calculate FPS
def calculate_fps(start_time, end_time):
    return 1 / (end_time - start_time) if (end_time - start_time) > 0 else 0

# Function to process frames and update the real-time charts
def process_frame(frame):
    start_time = time.time()

    # Object detection on modified frame
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


import random

# Dictionary to store random colors for each class
color_map = {}

# Function to generate a random color
def get_random_color():
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

# YOLOv5 video processor class for WebRTC live streaming
class YOLOv5VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.confidence_threshold = confidence_threshold  # Confidence threshold slider value from the sidebar
        self.class_selection = class_selection  # Class selection from the sidebar
        self.lock = threading.Lock()  # Ensure thread-safe access by adding a lock
        self.results = None
        self.current_class_count = {}
        self.frame_area = 0
        self.fps = 0
        self.inference_speed = 0

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")  # Get the video frame as a NumPy array

        # Process frame and collect metrics
        results, current_class_count, frame_area, fps, inference_speed = process_frame(img)

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
                class_name = model.names[int(cls)]
                if class_name in self.class_selection:
                    label = f'{class_name} {conf:.2f}'
                    cv2.rectangle(img, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
                    cv2.putText(img, label, (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Main logic to switch between detection modes (Live, Image, Video)
def main():
    # Layout and placement of different sections within a single frame with padding
    col1, col2, col3 = st.columns([1, 2, 1])

    with col1:
        st.markdown('<div class="section">', unsafe_allow_html=True)
        histogram_placeholder = st.empty()
        vehicle_pie_chart_placeholder = st.empty()
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section">', unsafe_allow_html=True)
        frame_placeholder = st.empty()  # Video frame placeholder
        pie_chart_placeholder = st.empty()
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="section">', unsafe_allow_html=True)
        traffic_graph_placeholder = st.empty()
        cumulative_graph_placeholder = st.empty()
        st.markdown('</div>', unsafe_allow_html=True)
    
    if detection_mode == "Live Detection":
        ctx = webrtc_streamer(
            key="object-detection",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=YOLOv5VideoProcessor,
            rtc_configuration={
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
            },
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

    # Loop for continuously updating metrics while webcam is running
        while ctx.state.playing:
            if ctx.video_processor:
                processor = ctx.video_processor
                if hasattr(processor, 'lock'):
                    with processor.lock:  # Ensure thread-safe access to the processor state
                        if processor.results:
                            # Calculate the metrics
                            total_objects = sum(processor.current_class_count.values())
                            fps_value = processor.fps
                            processed_frame = processor.results.render()[0]  # Get the processed frame
                            resolution_height, resolution_width = processed_frame.shape[:2]  # Extract height and width
                            inference_speed_value = processor.inference_speed

                            # Update the four colored blocks dynamically
                            update_metric_blocks(
                                total_objects, fps_value, resolution_width, resolution_height, inference_speed_value
                            )

                            rgb_frame = cv2.cvtColor(processor.results.render()[0], cv2.COLOR_BGR2RGB)
                            frame_placeholder.image(rgb_frame, use_column_width=True)


                            # Update charts below the video frame
                            histogram = update_histogram(processor.current_class_count)
                            pie_chart = calculate_area_proportions(processor.results, processor.frame_area)
                            vehicle_pie_chart = update_vehicle_proportion_chart(processor.current_class_count)

                            histogram_placeholder.plotly_chart(histogram, use_container_width=True)
                            pie_chart_placeholder.plotly_chart(pie_chart, use_container_width=True)
                            vehicle_pie_chart_placeholder.plotly_chart(vehicle_pie_chart, use_container_width=True)

                            traffic_graph, cumulative_graph = update_traffic_graph(timestamp_data)
                            traffic_graph_placeholder.plotly_chart(traffic_graph, use_container_width=True)
                            cumulative_graph_placeholder.plotly_chart(cumulative_graph, use_container_width=True)

            time.sleep(0.1)  # Small sleep to prevent resource overconsumption

    # Image upload detection mode
    elif detection_mode == "Upload Image":
        uploaded_image = st.sidebar.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
        if uploaded_image is not None:
            image = Image.open(uploaded_image)
            frame = np.array(image)
            
            results, current_class_count, frame_area, fps, inference_speed = process_frame(frame)
            
            total_objects = sum(current_class_count.values())
            fps_value = fps
            resolution_width = frame.shape[1]
            resolution_height = frame.shape[0]
            inference_speed_value = inference_speed

            # Update the four colored blocks dynamically
            update_metric_blocks(
                total_objects, fps_value, resolution_width, resolution_height, inference_speed_value
            )

            for *xyxy, conf, cls in results.xyxy[0]:
                if conf > confidence_threshold and model.names[int(cls)] in class_selection:
                    class_name = model.names[int(cls)]
                    label = f'{class_name} {conf:.2f}'
                    
                    # If this class doesn't have an assigned color yet, generate one
                    if class_name not in color_map:
                        color_map[class_name] = get_random_color()

                    # Get the color for the current class
                    color = color_map[class_name]
                    
                    # Draw bounding box with the selected color
                    cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), color, 2)
                    
                    # Put text label with the same color
                    cv2.putText(frame, label, (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Update the video frame below the metrics and above the graphs
            # Convert BGR to RGB before displaying the image
            # rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Ensure correct color conversion
            image_with_detections = Image.fromarray(frame)  # Convert back to PIL Image for display
            frame_placeholder.image(image_with_detections,use_column_width=True)

            histogram = update_histogram(current_class_count)
            pie_chart = calculate_area_proportions(results, frame_area)
            vehicle_pie_chart = update_vehicle_proportion_chart(current_class_count)

            histogram_placeholder.plotly_chart(histogram, use_container_width=True)
            pie_chart_placeholder.plotly_chart(pie_chart, use_container_width=True)
            vehicle_pie_chart_placeholder.plotly_chart(vehicle_pie_chart, use_container_width=True)

            traffic_graph, cumulative_graph = update_traffic_graph(timestamp_data)
            traffic_graph_placeholder.plotly_chart(traffic_graph, use_container_width=True)
            cumulative_graph_placeholder.plotly_chart(cumulative_graph, use_container_width=True)

    # Video upload detection mode
    elif detection_mode == "Upload Video":
        uploaded_video = st.sidebar.file_uploader("Upload a Video", type=["mp4", "avi", "mov"])
        # frame_placeholder = st.empty()
        
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

                total_objects = sum(current_class_count.values())
                fps_value = fps
                resolution_width = frame.shape[1]
                resolution_height = frame.shape[0]
                inference_speed_value = inference_speed

                # Update the four colored blocks dynamically
                update_metric_blocks(
                    total_objects, fps_value, resolution_width, resolution_height, inference_speed_value
                )

                for *xyxy, conf, cls in results.xyxy[0]:
                    if conf > confidence_threshold and model.names[int(cls)] in class_selection:
                        class_name = model.names[int(cls)]
                        label = f'{class_name} {conf:.2f}'
                        
                        # If this class doesn't have an assigned color yet, generate one
                        if class_name not in color_map:
                            color_map[class_name] = get_random_color()

                        # Get the color for the current class
                        color = color_map[class_name]
                        
                        # Draw bounding box with the selected color
                        cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), color, 2)
                        
                        # Put text label with the same color
                        cv2.putText(frame, label, (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                # Convert the frame from BGR to RGB after processing before rendering
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(rgb_frame, use_column_width=True)


                histogram = update_histogram(current_class_count)
                pie_chart = calculate_area_proportions(results, frame_area)
                vehicle_pie_chart = update_vehicle_proportion_chart(current_class_count)

                histogram_placeholder.plotly_chart(histogram, use_container_width=True)
                pie_chart_placeholder.plotly_chart(pie_chart, use_container_width=True)
                vehicle_pie_chart_placeholder.plotly_chart(vehicle_pie_chart, use_container_width=True)

                traffic_graph, cumulative_graph = update_traffic_graph(timestamp_data)
                traffic_graph_placeholder.plotly_chart(traffic_graph, use_container_width=True)
                cumulative_graph_placeholder.plotly_chart(cumulative_graph, use_container_width=True)

                time.sleep(0.03)

            cap.release()


if __name__ == "__main__":
    main()
