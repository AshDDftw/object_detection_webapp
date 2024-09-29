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

st.title("Object Detection Dashboard")

# Add custom CSS to reduce scrolling and provide better layout
st.markdown("""
    <style>
    .css-18e3th9 { padding-top: 1rem; padding-bottom: 1rem; }  /* Adjust padding */
    .css-1d391kg { padding-top: 1rem; padding-bottom: 1rem; }  /* Adjust padding */
    .stApp { overflow: hidden; }  /* Hide scrolling */
    </style>
    """, unsafe_allow_html=True)

# Sidebar for options
st.sidebar.title("Settings")

# Option for class-wise selection
class_selection = st.sidebar.multiselect(
    "Select Classes to Detect",
    options=list(model.names.values()),  # Ensure it's a list
    # default=list(model.names.values())   # Default to all classes
)

# Option to select static image/video upload or live detection
detection_mode = st.sidebar.radio(
    "Detection Mode",
    ("Live Detection", "Upload Image", "Upload Video")
)

# Add weather dropdown to the sidebar
weather_conditions = st.sidebar.selectbox(
    "Select Weather Condition",
    ("Normal", "Rainy", "Cloudy", "Foggy")
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

# Function to add rain effect on the image
def add_rain(image):
    rain_overlay = image.copy()
    h, w, _ = image.shape
    rain_drops = np.zeros_like(image, dtype=np.uint8)
    for i in range(1000):  # Number of raindrops
        x = np.random.randint(0, w)
        y = np.random.randint(0, h)
        rain_drops = cv2.line(rain_drops, (x, y), (x, y + np.random.randint(5, 20)), (200, 200, 200), 1)
    
    rain_overlay = cv2.addWeighted(rain_overlay, 0.8, rain_drops, 0.2, 0)
    return rain_overlay

# Function to add fog effect on the image
def add_fog(image):
    fog_overlay = image.copy()
    fog = np.zeros_like(image, dtype=np.uint8)
    fog = cv2.circle(fog, (fog.shape[1] // 2, fog.shape[0] // 2), fog.shape[1] // 3, (255, 255, 255), -1)
    fog = cv2.GaussianBlur(fog, (61, 61), 0)
    fog_overlay = cv2.addWeighted(fog_overlay, 0.8, fog, 0.2, 0)
    return fog_overlay

# Function to add cloudy effect on the image
def add_cloud(image):
    cloud_overlay = image.copy()
    clouds = np.zeros_like(image, dtype=np.uint8)
    for i in range(10):  # Number of clouds
        x, y = np.random.randint(0, cloud_overlay.shape[1]), np.random.randint(0, cloud_overlay.shape[0] // 2)
        radius = np.random.randint(50, 150)
        clouds = cv2.circle(clouds, (x, y), radius, (200, 200, 200), -1)
    clouds = cv2.GaussianBlur(clouds, (101, 101), 0)
    cloud_overlay = cv2.addWeighted(cloud_overlay, 0.7, clouds, 0.3, 0)
    return cloud_overlay

# Update the traffic graph and remove initial date on the x-axis
def update_traffic_graph(timestamp_data):
    df = pd.DataFrame(timestamp_data)
    
    df['time'] = pd.to_datetime(df['time'], format='%H:%M:%S')
    
    df_second = df.groupby(df['time'].dt.floor('S')).size().reset_index(name='count_per_second')
    df_second['cumulative_count'] = df_second['count_per_second'].cumsum()

    # Formatting to only show time without the date
    df_second['time'] = df_second['time'].dt.strftime('%H:%M:%S')

    traffic_graph = px.line(df_second, x='time', y='count_per_second', title="Traffic Per Second", template="plotly_dark", height=250)
    cumulative_graph = px.line(df_second, x='time', y='cumulative_count', title="Cumulative Traffic", template="plotly_dark", height=250)

    return traffic_graph, cumulative_graph

# Define the function to update the vehicle proportion pie chart
def update_vehicle_proportion_chart(class_count):
    vehicle_classes = ['car', 'truck', 'bus']  # You can adjust the vehicle classes
    vehicle_count = {cls: class_count[cls] for cls in vehicle_classes if cls in class_count and class_count[cls] > 0}
    
    if vehicle_count:
        vehicle_data = {"Vehicle Type": list(vehicle_count.keys()), "Count": list(vehicle_count.values())}
        df = pd.DataFrame(vehicle_data)
        fig = px.pie(df, names="Vehicle Type", values="Count", title="Vehicle Proportion",
                     template="plotly_dark", color_discrete_sequence=px.colors.sequential.RdBu, height=250)
    else:
        fig = px.pie(title="No Vehicles Detected", template="plotly_dark", height=250)
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

# Define the function to update the timestamp line graph
def update_timestamp_line_graph(timestamp_data):
    df = pd.DataFrame(timestamp_data)
    if not df.empty:
        fig = px.line(df, x="time", y="count", color="class", title="Detection Over Time",
                      template="plotly_dark", height=250)
    else:
        fig = px.line(title="No Data to Display Yet", template="plotly_dark", height=250)
        fig.update_layout(xaxis={'visible': False}, yaxis={'visible': False}, annotations=[{
            'text': 'No Objects Detected Yet',
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
                     template="plotly_dark", color_discrete_sequence=px.colors.sequential.RdBu, height=250)
    else:
        fig = px.pie(title="No Objects Detected", template="plotly_dark", height=250)
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

    # Apply selected weather effect
    if weather_conditions == "Rainy":
        frame = add_rain(frame)
    elif weather_conditions == "Cloudy":
        frame = add_cloud(frame)
    elif weather_conditions == "Foggy":
        frame = add_fog(frame)

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
    col1, col2, col3 = st.columns([1, 2, 1])

    with col1:
        histogram_placeholder = st.empty()
        vehicle_pie_chart_placeholder = st.empty()
    
    with col2:
        pie_chart_placeholder = st.empty()
    
    with col3:
        traffic_graph_placeholder = st.empty()
        cumulative_graph_placeholder = st.empty()
    
    # WebRTC Live Detection mode
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

        # Create a placeholder outside the loop
        frame_placeholder = st.empty()

        # Loop for continuously updating metrics while webcam is running
        while ctx.state.playing:
            if ctx.video_processor:
                processor = ctx.video_processor
                if hasattr(processor, 'lock'):
                    with processor.lock:  # Ensure thread-safe access to the processor state
                        if processor.results:
                            total = sum(processor.current_class_count.values())
                            total_objects.metric("Total Objects", total)
                            fps_display.metric("FPS", f"{processor.fps:.2f}")
                            resolution_display.metric("Resolution", f"{640}x{480}")
                            inference_speed_display.metric("Inference Speed (ms)", f"{processor.inference_speed:.2f}")

                            # Update charts
                            histogram = update_histogram(processor.current_class_count)
                            pie_chart = calculate_area_proportions(processor.results, processor.frame_area)
                            vehicle_pie_chart = update_vehicle_proportion_chart(processor.current_class_count)

                            histogram_placeholder.plotly_chart(histogram, use_container_width=True)
                            pie_chart_placeholder.plotly_chart(pie_chart, use_container_width=True)
                            vehicle_pie_chart_placeholder.plotly_chart(vehicle_pie_chart, use_container_width=True)

                            traffic_graph, cumulative_graph = update_traffic_graph(timestamp_data)
                            traffic_graph_placeholder.plotly_chart(traffic_graph, use_container_width=True)
                            cumulative_graph_placeholder.plotly_chart(cumulative_graph, use_container_width=True)

                            # Update the video frame in the same placeholder
                            frame_placeholder.image(processor.results.render()[0], caption="Video Frame", use_column_width=True)

            time.sleep(0.1)  # Small sleep to prevent resource overconsumption


    # Image upload detection mode
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

    # Video upload detection mode
    elif detection_mode == "Upload Video":
        uploaded_video = st.sidebar.file_uploader("Upload a Video", type=["mp4", "avi", "mov"])
        frame_placeholder = st.empty()
        
        if uploaded_video is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_video.read())
            
            cap = cv2.VideoCapture(tfile.name)

            # Create a placeholder for the video frames
            frame_placeholder = st.empty()

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
                        cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
                        cv2.putText(frame, label, (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(rgb_frame)

                # Update the video frame in the same placeholder
                frame_placeholder.image(image, caption="Video Frame", use_column_width=True)

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
