import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase
from ultralytics import YOLO
import cv2
import numpy as np
import threading
from datetime import datetime, timedelta
import time
import plotly.express as px
import av  # Ensure that av is imported correctly
from app_utils import update_histogram, calculate_area_proportions, update_vehicle_proportion_chart, update_traffic_graph, apply_weather_effect


# Global variable to store traffic data (last 10 seconds)
traffic_data = []

# Load the YOLOv8 model (you can change the path or model variant if needed)
model = YOLO('yolov8n.pt')


# YOLOv8VideoProcessor class definition to process video frames
class YOLOv8VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.confidence_threshold = 0.5
        self.class_selection = []
        self.weather_conditions = 'Normal'
        self.lock = threading.Lock()  # Lock to manage thread safety
        self.results = None
        self.current_class_count = {}
        self.fps = 0
        self.inference_speed = 0
        self.frame_area = 0

    def update_params(self, confidence_threshold, class_selection, weather_conditions):
        with self.lock:
            self.confidence_threshold = confidence_threshold
            self.class_selection = class_selection
            self.weather_conditions = weather_conditions

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")  # Convert frame to NumPy array (BGR format)

        # Apply the weather effect to the frame
        processed_frame = apply_weather_effect(img, self.weather_conditions)

        start_time = time.time()

        # YOLOv8 inference
        results = model(processed_frame)
        
        end_time = time.time()
        self.results = results

        # Calculate frame area and fps
        self.frame_area = processed_frame.shape[0] * processed_frame.shape[1]
        self.fps = 1 / (end_time - start_time) if (end_time - start_time) > 0 else 0
        self.inference_speed = (end_time - start_time) * 1000  # Convert to milliseconds

        # Process detected objects and update the class count
        class_count = {name: 0 for name in model.names.values()}
        for box in results[0].boxes:
            conf = box.conf.item()
            cls = int(box.cls.item())

            if conf > self.confidence_threshold:
                class_name = model.names[cls]
                if class_name in self.class_selection:
                    class_count[class_name] += 1

        self.current_class_count = class_count

        # Draw bounding boxes on the frame
        for box in results[0].boxes:
            xyxy = box.xyxy[0].cpu().numpy()  # Bounding box coordinates
            conf = box.conf.item()
            cls = int(box.cls.item())

            if conf > self.confidence_threshold:
                class_name = model.names[cls]
                if class_name in self.class_selection:
                    label = f'{class_name} {conf:.2f}'
                    # Draw bounding box and label
                    cv2.rectangle(processed_frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
                    cv2.putText(processed_frame, label, (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        return av.VideoFrame.from_ndarray(processed_frame, format="bgr24")


# Live detection function
def run_live_detection(
    confidence_threshold, 
    class_selection, 
    frame_placeholder, 
    pie_chart_placeholder_1, 
    pie_chart_placeholder_2, 
    traffic_graph_placeholder, 
    cumulative_graph_placeholder,
    class_distribution_placeholder,
    update_metric_blocks,  # Function to update metric blocks dynamically
    weather_conditions
):
    # Create a WebRTC streamer instance for live detection
    ctx = webrtc_streamer(
        key="object-detection",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=YOLOv8VideoProcessor,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    # Loop for continuously updating metrics while the webcam is running
    while ctx.state.playing:
        if ctx.video_processor:
            processor = ctx.video_processor

            # Dynamically update the confidence threshold, class selection, and weather conditions
            processor.update_params(confidence_threshold, class_selection, weather_conditions)

            if hasattr(processor, 'lock'):
                with processor.lock:
                    if processor.results:
                        # Get the processed frame and metrics
                        total_objects = sum(processor.current_class_count.values())
                        fps_value = processor.fps

                        # Use the processed frame (directly from the video processor)
                        processed_frame = processor.results[0].orig_img  # Use original processed image from results

                        resolution_height, resolution_width = processed_frame.shape[:2]
                        inference_speed_value = processor.inference_speed

                        # Update the metric blocks
                        update_metric_blocks(
                            total_objects, fps_value, resolution_width, resolution_height, inference_speed_value
                        )

                        # Convert the processed frame to RGB and display it
                        rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                        frame_placeholder.image(rgb_frame, use_column_width=True)

                        # Update charts below the video frame
                        histogram = update_histogram(processor.current_class_count) or px.bar(
                            title="Waiting for Object Detection...", template="plotly_dark", height=250)
                        
                        pie_chart_1 = calculate_area_proportions(processor.results, processor.frame_area, class_selection) or px.pie(
                            title="No Objects Detected", template="plotly_dark", height=290)
                        
                        pie_chart_2 = update_vehicle_proportion_chart(processor.current_class_count) or px.pie(
                            title="No Vehicles Detected", template="plotly_dark", height=290)

                        pie_chart_placeholder_1.plotly_chart(pie_chart_1, use_container_width=True)
                        pie_chart_placeholder_2.plotly_chart(pie_chart_2, use_container_width=True)

                        # Update traffic data for the sliding window of the last 10 seconds
                        global traffic_data
                        current_time = datetime.now().strftime('%H:%M:%S')

                        # Append the new timestamp data
                        for class_name, count in processor.current_class_count.items():
                            traffic_data.append({
                                "time": current_time,
                                "class": class_name,
                                "count": count
                            })

                        # Keep only the last 10 seconds of data
                        traffic_data = [data for data in traffic_data if data["time"] >= (datetime.now() - timedelta(seconds=10)).strftime('%H:%M:%S')]

                        # Pass the sliding window traffic data to the update_traffic_graph function
                        traffic_graph, cumulative_graph = update_traffic_graph(traffic_data)

                        # Check if graphs are returned before plotting them
                        if traffic_graph:
                            traffic_graph_placeholder.plotly_chart(traffic_graph, use_container_width=True)
                        else:
                            traffic_graph_placeholder.plotly_chart(px.line(
                                title="No Traffic Detected", template="plotly_dark", height=250), use_container_width=True)

                        if cumulative_graph:
                            cumulative_graph_placeholder.plotly_chart(cumulative_graph, use_container_width=True)
                        else:
                            cumulative_graph_placeholder.plotly_chart(px.line(
                                title="No Cumulative Traffic", template="plotly_dark", height=250), use_container_width=True)

                        # Update class distribution histogram
                        class_distribution_placeholder.plotly_chart(histogram, use_container_width=True)

            time.sleep(0.1)  # Small sleep to prevent resource overconsumption
