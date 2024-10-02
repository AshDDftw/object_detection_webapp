import streamlit as st
from PIL import Image
import numpy as np
import config
from app_utils import process_frame, update_histogram, calculate_area_proportions, update_vehicle_proportion_chart, update_traffic_graph, get_random_color
import cv2
from datetime import datetime , timedelta


traffic_data = []

# Function to handle image upload detection
def run_image_detection(
    confidence_threshold,  # Pass confidence threshold
    class_selection,       # Pass selected classes
    frame_placeholder, 
    pie_chart_placeholder_1, 
    pie_chart_placeholder_2, 
    traffic_graph_placeholder, 
    cumulative_graph_placeholder,
    class_distribution_placeholder,
    update_metric_blocks   # Function to update metric blocks dynamically
):
    uploaded_image = st.sidebar.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        frame = np.array(image)

        # Process frame using the model from config
        results, current_class_count, frame_area, fps, inference_speed, timestamp_data = process_frame(frame, config.model, confidence_threshold, class_selection)

        total_objects = sum(current_class_count.values())
        resolution_width = frame.shape[1]
        resolution_height = frame.shape[0]

        # Update the metric blocks dynamically
        update_metric_blocks(total_objects, fps, resolution_width, resolution_height, inference_speed)

        color_map = {}  # Dictionary to store random colors for each class

        # Draw bounding boxes with class labels and colors
        for *xyxy, conf, cls in results.xyxy[0]:
            if conf > confidence_threshold and config.model.names[int(cls)] in class_selection:
                class_name = config.model.names[int(cls)]
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

        # Convert BGR to RGB and display the image
        image_with_detections = Image.fromarray(frame)
        frame_placeholder.image(image_with_detections, use_column_width=True)

        # Update the traffic_data list with new timestamps and class counts
        global traffic_data

        # Append new timestamp data
        for i, cls in enumerate(timestamp_data["class"]):
            traffic_data.append({
                "time": timestamp_data["time"][i],
                "class": cls,
                "count": timestamp_data["count"][i]
            })

        # Keep only the last 10 seconds of data (sliding window logic)
        current_time = datetime.now().strftime('%H:%M:%S')
        traffic_data = [data for data in traffic_data if data["time"] >= (datetime.now() - timedelta(seconds=10)).strftime('%H:%M:%S')]

        # Update histogram, pie charts, and traffic graphs
        histogram = update_histogram(current_class_count)
        # Assuming you have class_selection in your main app logic
        pie_chart_1 = calculate_area_proportions(results, frame_area, class_selection)

        pie_chart_2 = update_vehicle_proportion_chart(current_class_count)

        pie_chart_placeholder_1.plotly_chart(pie_chart_1, use_container_width=True)
        pie_chart_placeholder_2.plotly_chart(pie_chart_2, use_container_width=True)

        # Pass the sliding window traffic data to the update_traffic_graph function
        traffic_graph, cumulative_graph = update_traffic_graph(traffic_data)
        
        if traffic_graph and cumulative_graph:
            traffic_graph_placeholder.plotly_chart(traffic_graph, use_container_width=True)
            cumulative_graph_placeholder.plotly_chart(cumulative_graph, use_container_width=True)

        class_distribution_placeholder.plotly_chart(histogram, use_container_width=True)