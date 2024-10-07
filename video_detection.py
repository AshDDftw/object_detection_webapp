import streamlit as st
import tempfile
import cv2
import config
from app_utils import process_frame, update_histogram, calculate_area_proportions, update_vehicle_proportion_chart, update_traffic_graph, get_random_color,apply_weather_effect
import time
from datetime import datetime , timedelta

# Initialize global variable to store traffic data (last 10 seconds)
traffic_data = []

import plotly.express as px  # Ensure Plotly Express is imported for graph generation

def run_video_detection(
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

            # Apply weather effect
            weather_frame = apply_weather_effect(frame, weather_conditions)

            # Process the frame using the model from config
            results, current_class_count, frame_area, fps, inference_speed, timestamp_data = process_frame(
                weather_frame, config.model, confidence_threshold, class_selection
            )

            # Handle frame display and object detection results
            if results is not None and results[0].boxes is not None:
                for box in results[0].boxes:
                    xyxy = box.xyxy[0].cpu().numpy()
                    conf = box.conf.item()
                    cls = int(box.cls.item())

                    if conf > confidence_threshold:
                        class_name = config.model.names[cls]
                        if class_name in class_selection:
                            label = f'{class_name} {conf:.2f}'
                            cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
                            cv2.putText(frame, label, (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # Convert the frame from BGR to RGB for display
            rgb_frame = cv2.cvtColor(weather_frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(rgb_frame, use_column_width=True)

            # Update traffic data for the last 10 seconds
            global traffic_data

            # Append new timestamp data
            for i, cls in enumerate(timestamp_data["class"]):
                traffic_data.append({
                    "time": timestamp_data["time"][i],
                    "class": cls,
                    "count": timestamp_data["count"][i]
                })

            # Keep only the last 10 seconds of data
            traffic_data = [data for data in traffic_data if data["time"] >= (datetime.now() - timedelta(seconds=10)).strftime('%H:%M:%S')]

            # Update the metrics: total objects, FPS, resolution, inference speed
            total_objects = sum(current_class_count.values())
            update_metric_blocks(
                total_objects=total_objects,
                fps_value=fps,
                resolution_width=frame.shape[1],
                resolution_height=frame.shape[0],
                inference_speed_value=inference_speed
            )

            # Update the graphs even if no objects were detected
            histogram = update_histogram(current_class_count)

            # Check if results are not None before calculating area proportions
            if results is not None and results[0].boxes is not None:
                pie_chart_1 = calculate_area_proportions(results, frame_area, class_selection)
            else:
                pie_chart_1 = px.pie(title="No Objects Detected", template="plotly_dark", height=290)

            pie_chart_2 = update_vehicle_proportion_chart(current_class_count)
            if pie_chart_2 is None:
                pie_chart_2 = px.pie(title="No Vehicles Detected", template="plotly_dark", height=290)

            # Display pie charts
            pie_chart_placeholder_1.plotly_chart(pie_chart_1, use_container_width=True)
            pie_chart_placeholder_2.plotly_chart(pie_chart_2, use_container_width=True)

            # Update traffic and cumulative graphs
            traffic_graph, cumulative_graph = update_traffic_graph(traffic_data)

            # Display traffic graphs
            if traffic_graph is None:
                traffic_graph = px.line(title="No Traffic Detected", template="plotly_dark", height=250)
            if cumulative_graph is None:
                cumulative_graph = px.line(title="No Cumulative Traffic Detected", template="plotly_dark", height=250)

            traffic_graph_placeholder.plotly_chart(traffic_graph, use_container_width=True)
            cumulative_graph_placeholder.plotly_chart(cumulative_graph, use_container_width=True)

            # Ensure class distribution histogram is shown, even if no objects are detected
            if histogram is None:
                histogram = px.bar(title="Waiting for Object Detection...", template="plotly_dark", height=250)

            class_distribution_placeholder.plotly_chart(histogram, use_container_width=True)

            time.sleep(0.1)

        cap.release()
