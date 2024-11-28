import cv2
import numpy as np
import streamlit as st
from PIL import Image
from datetime import datetime, timedelta
from app_utils import process_frame, update_histogram, calculate_area_proportions, update_object_proportion_chart, update_traffic_graph, apply_weather_effect,generate_heatmap
import config

traffic_data = []


def run_image_detection(
    confidence_threshold, 
        class_selection, 
        frame_placeholder, 
        pie_chart_placeholder_1, 
        pie_chart_placeholder_2, 
        traffic_graph_placeholder, 
        cumulative_graph_placeholder,
        class_distribution_placeholder,
        update_metric_blocks,
        weather_conditions
):
    uploaded_image = st.sidebar.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        frame = np.array(image)

        frame = cv2.resize(frame, (1280, 1200) , interpolation=cv2.INTER_AREA)

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

        # Convert BGR to RGB and display the original frame with detections
        image_with_detections = Image.fromarray(weather_frame)
        frame_placeholder.image(image_with_detections, use_column_width=True)

        # Update the traffic_data list with new timestamps and class counts
        global traffic_data
        for i, cls in enumerate(timestamp_data["class"]):
            traffic_data.append({
                "time": timestamp_data["time"][i],
                "class": cls,
                "count": timestamp_data["count"][i]
            })

        # Keep only the last 10 seconds of data
        traffic_data = [data for data in traffic_data if data["time"] >= (datetime.now() - timedelta(seconds=10)).strftime('%H:%M:%S')]


        traffic_graph,_ = update_traffic_graph(traffic_data,class_selection)
        # Update the graphs even if no objects were detected
        histogram = update_histogram(results, class_selection)
        class_distribution_placeholder.plotly_chart(histogram, use_container_width=True)

        # Update pie charts
        pie_chart_1 = calculate_area_proportions(results, frame_area, class_selection)
        pie_chart_2 = update_object_proportion_chart(results, class_selection)

        # Display the pie charts
        pie_chart_placeholder_1.plotly_chart(pie_chart_1, use_container_width=True)
        pie_chart_placeholder_2.plotly_chart(pie_chart_2, use_container_width=True)

        # Replace cumulative graph with heatmap visualization
        heatmap_overlay = generate_heatmap(weather_frame, results)
        heatmap_image = Image.fromarray(heatmap_overlay)
        traffic_graph_placeholder.plotly_chart(traffic_graph, use_container_width=True)
        # cumulative_graph_placeholder.image(heatmap_image, use_column_width=True)
