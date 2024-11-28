import cv2
import numpy as np
import streamlit as st
from datetime import datetime, timedelta
from app_utils import process_frame, update_histogram, calculate_area_proportions, update_object_proportion_chart, update_traffic_graph, apply_weather_effect,generate_heatmap
import time
import tempfile
import config

# Initialize global variable to store traffic data (last 10 seconds)
traffic_data = []


# Function to run video detection and generate heatmaps for each frame
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

            frame = cv2.resize(frame, (1280, 1200))

            # Apply weather effect to the frame
            weather_frame = apply_weather_effect(frame, weather_conditions)

            # Process the frame using the model from config
            results, current_class_count, frame_area, fps, inference_speed, timestamp_data = process_frame(
                weather_frame, config.model, confidence_threshold, class_selection
            )

            # Update metrics dynamically
            total_objects = sum(current_class_count.values())
            resolution_width = weather_frame.shape[1]
            resolution_height = weather_frame.shape[0]
            update_metric_blocks(total_objects, fps, resolution_width, resolution_height, inference_speed)

            # Convert the frame from BGR to RGB for display
            rgb_frame = cv2.cvtColor(weather_frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(rgb_frame, use_column_width=True)

            # Generate the heatmap for the frame
            heatmap_overlay = generate_heatmap(weather_frame, results)

            

            # Update the traffic_data list with new timestamps and class counts
            global traffic_data
            for i, cls in enumerate(timestamp_data["class"]):
                # Get the current time
                current_time = timestamp_data["time"][i]
                found = False
                # Check if there's already an entry for the current time and class
                for data in traffic_data:
                    if data["time"] == current_time and data["class"] == cls:
                        data["count"] += timestamp_data["count"][i]  # Update the count
                        found = True
                        break
                if not found:
                    # If no matching time and class found, append a new entry
                    traffic_data.append({
                        "time": current_time,
                        "class": cls,
                        "count": timestamp_data["count"][i]
                    })

            # Keep only the last 10 seconds of data
            traffic_data = [data for data in traffic_data if data["time"] >= (datetime.now() - timedelta(seconds=10)).strftime('%H:%M:%S')]

            # Update the graphs even if no objects were detected
            histogram = update_histogram(results, class_selection)

            # Calculate area proportions for pie charts
            pie_chart_1 = calculate_area_proportions(results, frame_area, class_selection)
            pie_chart_2 = update_object_proportion_chart(results, class_selection)

            # Display the pie charts
            pie_chart_placeholder_1.plotly_chart(pie_chart_1, use_container_width=True)
            pie_chart_placeholder_2.plotly_chart(pie_chart_2, use_container_width=True)

            # Update traffic and cumulative graphs
            traffic_graph, cumulative_graph = update_traffic_graph(traffic_data, class_selection)

            # Display traffic and cumulative graphs
            traffic_graph_placeholder.plotly_chart(traffic_graph, use_container_width=True)
            # cumulative_graph_placeholder.image(heatmap_overlay, use_column_width=True)

            # Ensure class distribution histogram is shown
            class_distribution_placeholder.plotly_chart(histogram, use_container_width=True)

            time.sleep(1)  # Ensure data updates every second

        cap.release()
