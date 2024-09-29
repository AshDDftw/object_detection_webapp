import cv2
import streamlit as st
from PIL import Image
import numpy as np
from common import (
    process_frame,
    update_histogram,
    calculate_area_proportions,
    update_vehicle_proportion_chart,
    update_traffic_graph,
    update_timestamp_line_graph,
    total_objects,
    fps_display,
    resolution_display,
    inference_speed_display,
    timestamp_data,
)

def run_image_detection():
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

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_with_detections = Image.fromarray(rgb_frame)
        st.image(image_with_detections, caption="Uploaded Image", use_column_width=True)

        histogram = update_histogram(current_class_count)
        pie_chart = calculate_area_proportions(results, frame_area)
        vehicle_pie_chart = update_vehicle_proportion_chart(current_class_count)

        st.plotly_chart(histogram, use_container_width=True)
        st.plotly_chart(pie_chart, use_container_width=True)
        st.plotly_chart(vehicle_pie_chart, use_container_width=True)

        traffic_graph, cumulative_graph = update_traffic_graph(timestamp_data)
        st.plotly_chart(traffic_graph, use_container_width=True)
        st.plotly_chart(cumulative_graph, use_container_width=True)

        line_graph = update_timestamp_line_graph(timestamp_data)
        st.plotly_chart(line_graph, use_container_width=True)
