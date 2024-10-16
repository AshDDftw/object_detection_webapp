import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase
from ultralytics import YOLO
import cv2
import numpy as np
import threading
from datetime import datetime, timedelta
import time
import plotly.express as px
import av
import pandas as pd
import config
from app_utils import update_histogram, calculate_area_proportions, update_vehicle_proportion_chart, update_traffic_graph, apply_weather_effect,generate_heatmap

# Global variable to store traffic data (last 10 seconds)
traffic_data = []


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
        results = config.model(processed_frame)
        
        end_time = time.time()
        self.results = results

        obb_results = getattr(self.results[0], 'obb', None)  # Safely get OBB results, might be None

        # Calculate frame area and fps
        self.frame_area = processed_frame.shape[0] * processed_frame.shape[1]
        self.fps = 1 / (end_time - start_time) if (end_time - start_time) > 0 else 0
        self.inference_speed = (end_time - start_time) * 1000  # Convert to milliseconds

        # Process detected objects and update the class count
        class_count = {name: 0 for name in config.model.names.values()}
        
        if obb_results is not None:
            for i in range(obb_results.shape[0]):
                xywhr = obb_results.data[i][:5]  # Get the xywhr coordinates
                conf = obb_results.data[i][5].item()  # Get the confidence score
                cls_idx = int(obb_results.data[i][6].item())  # Get the class index

                # Filter results based on confidence threshold and selected classes
                if conf >= self.confidence_threshold:
                    class_name = config.model.names[cls_idx]
                    if class_name in self.class_selection:
                        # Increment the class count
                        class_count[class_name] += 1

                        # Draw bounding box on the frame (you can use xyxyxyxy for an 8-point polygon)
                        xyxyxyxy = obb_results.xyxyxyxy[i]  # 8-point polygon coordinates

                        # Convert to integer coordinates for drawing
                        points = [(int(x), int(y)) for x, y in xyxyxyxy]

                        # Draw the oriented bounding box on the frame
                        cv2.polylines(processed_frame, [np.array(points)], isClosed=True, color=(0, 255, 0), thickness=2)

                        # Put label and confidence score near the bounding box
                        label = f'{class_name} {conf:.2f}'
                        cv2.putText(processed_frame, label, (points[0][0], points[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Generate the heatmap for the frame and overlay it on the frame
        overlaid_frame = generate_heatmap(processed_frame, self.results)

        return av.VideoFrame.from_ndarray(overlaid_frame, format="bgr24")

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
                        histogram = update_histogram(processor.results, class_selection) or px.bar(
                            title="Waiting for Object Detection...", template="plotly_dark", height=250)
                        
                        pie_chart_1 = calculate_area_proportions(processor.results, processor.frame_area, class_selection) or px.pie(
                            title="No Objects Detected", template="plotly_dark", height=290)
                        
                        pie_chart_2 = update_vehicle_proportion_chart(processor.results, class_selection) or px.pie(
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
                        traffic_graph, _ = update_traffic_graph(traffic_data, class_selection)

                        # Check if graphs are returned before plotting them
                        if traffic_graph:
                            traffic_graph_placeholder.plotly_chart(traffic_graph, use_container_width=True)
                        else:
                            traffic_graph_placeholder.plotly_chart(px.line(
                                title="No Traffic Detected", template="plotly_dark", height=250), use_container_width=True)

                        # Display the heatmap in the cumulative graph placeholder
                        heatmap_frame = generate_heatmap(processed_frame, processor.results)
                        cumulative_graph_placeholder.image(heatmap_frame, use_column_width=True)

                        # Update class distribution histogram
                        class_distribution_placeholder.plotly_chart(histogram, use_container_width=True)

            time.sleep(0.1)  # Small sleep to prevent resource overconsumption
