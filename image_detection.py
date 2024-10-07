import streamlit as st
from PIL import Image
import numpy as np
import config
from app_utils import process_frame, update_histogram, calculate_area_proportions, update_vehicle_proportion_chart, update_traffic_graph, get_random_color, apply_weather_effect
import cv2
from datetime import datetime, timedelta
import plotly.express as px 

traffic_data = []

def run_image_detection(
    confidence_threshold,  # Pass confidence threshold
    class_selection,       # Pass selected classes
    frame_placeholder, 
    pie_chart_placeholder_1, 
    pie_chart_placeholder_2, 
    traffic_graph_placeholder, 
    cumulative_graph_placeholder,
    class_distribution_placeholder,
    update_metric_blocks,   # Function to update metric blocks dynamically
    weather_conditions
):
    uploaded_image = st.sidebar.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        frame = np.array(image)

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

        # Draw bounding boxes if there are results
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

        # Convert BGR to RGB and display the image
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
        current_time = datetime.now().strftime('%H:%M:%S')
        traffic_data = [data for data in traffic_data if data["time"] >= (datetime.now() - timedelta(seconds=10)).strftime('%H:%M:%S')]

        # Update the graphs even if no objects were detected
        histogram = update_histogram(current_class_count)
        if histogram is None:
            histogram = px.bar(title="Waiting for Object Detection...", template="plotly_dark", height=250)

        # Display the histogram
        class_distribution_placeholder.plotly_chart(histogram, use_container_width=True)

        # Check if results are not None before calculating area proportions
        if results is not None and results[0].boxes is not None:
            pie_chart_1 = calculate_area_proportions(results, frame_area, class_selection)
        else:
            pie_chart_1 = px.pie(title="No Objects Detected", template="plotly_dark", height=290)

        pie_chart_2 = update_vehicle_proportion_chart(current_class_count)
        if pie_chart_2 is None:
            pie_chart_2 = px.pie(title="No Vehicles Detected", template="plotly_dark", height=290)

        # Display the pie charts
        pie_chart_placeholder_1.plotly_chart(pie_chart_1, use_container_width=True)
        pie_chart_placeholder_2.plotly_chart(pie_chart_2, use_container_width=True)

        # Update traffic and cumulative graphs
        traffic_graph, cumulative_graph = update_traffic_graph(traffic_data)
        if traffic_graph is None:
            traffic_graph = px.line(title="No Traffic Detected", template="plotly_dark", height=250)
        if cumulative_graph is None:
            cumulative_graph = px.line(title="No Cumulative Traffic", template="plotly_dark", height=250)

        # Display traffic and cumulative graphs
        traffic_graph_placeholder.plotly_chart(traffic_graph, use_container_width=True)
        cumulative_graph_placeholder.plotly_chart(cumulative_graph, use_container_width=True)
