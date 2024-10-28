import time
import cv2
import torch
import random
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
from streamlit_webrtc import VideoProcessorBase
import threading
import av
import streamlit as st
import config
import imgaug.augmenters as iaa
import numpy as np

# Global variables to track traffic data
timestamp_data = {"time": [], "class": [], "count": []}
color_map = {}


def process_frame(frame, model, confidence_threshold, class_selection):
    start_time = time.time()

    # Perform object detection using YOLOv8 model and get OBB results
    results = model.predict(frame)
    obb_results = results[0].obb  # Extract oriented bounding boxes (OBBs)

    # Measure inference time and FPS
    end_time = time.time()
    inference_speed = (end_time - start_time) * 1000  # Convert to milliseconds
    fps = 1 / (end_time - start_time) if (end_time - start_time) > 0 else 0  # FPS

    frame_area = frame.shape[0] * frame.shape[1]  # Calculate frame area

    # Initialize class count and timestamp data
    class_count = {name: 0 for name in model.names.values()}
    timestamp_data = {"time": [], "class": [], "count": []}

    current_time = datetime.now().strftime('%H:%M:%S')

    # Check if there are any boxes detected
    if obb_results is None or len(obb_results) == 0:
        # No detections, return empty counts
        timestamp_data["time"].append(current_time)
        timestamp_data["class"].append("None")
        timestamp_data["count"].append(0)
        return frame, class_count, frame_area, fps, inference_speed, timestamp_data

    # Loop through each detected box and process the results
    for i in range(obb_results.shape[0]):
        xywhr = obb_results.data[i][:5]  # Get the xywhr coordinates
        conf = obb_results.data[i][5].item()  # Get the confidence score
        cls_idx = int(obb_results.data[i][6].item())  # Get the class index

        # Filter results based on confidence threshold and selected classes
        if conf >= confidence_threshold:
            class_name = model.names[cls_idx]
            if class_name in class_selection:
                # Increment the class count
                class_count[class_name] += 1

                # Add timestamp data for detected objects
                timestamp_data["time"].append(current_time)
                timestamp_data["class"].append(class_name)
                timestamp_data["count"].append(class_count[class_name])

                # Draw bounding box on the frame (you can use xyxyxyxy for an 8-point polygon)
                xyxyxyxy = obb_results.xyxyxyxy[i]  # 8-point polygon coordinates

                # Convert to integer coordinates for drawing
                points = [(int(x), int(y)) for x, y in xyxyxyxy]

                # Draw the oriented bounding box on the frame
                cv2.polylines(frame, [np.array(points)], isClosed=True, color=(0, 255, 0), thickness=2)

                # Put label and confidence score near the bounding box
                label = f'{class_name} {conf:.2f}'
                cv2.putText(frame, label, (points[0][0], points[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Return the annotated frame, class count, frame area, FPS, inference speed, and timestamp data
    return results, class_count, frame_area, fps, inference_speed, timestamp_data


def update_metric_blocks(total_objects, fps_value, resolution_width, resolution_height, inference_speed_value):
    st.markdown(f"""
        <div style="background-color: #f94144; padding: 10px; border-radius: 10px;margin:10px color: white; text-align: center;">
            <h3>Total Objects</h3>
            <h2>{total_objects}</h2>
        </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
        <div style="background-color: #f3722c; padding: 10px; border-radius: 10px; color: white; text-align: center;">
            <h3>FPS</h3>
            <h2>{fps_value:.2f}</h2>
        </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
        <div style="background-color: #f9c74f; padding: 10px; border-radius: 10px; color: black; text-align: center;">
            <h3>Resolution</h3>
            <h2>{resolution_width}x{resolution_height}</h2>
        </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
        <div style="background-color: #43aa8b; padding: 10px; border-radius: 10px;margin-bottom:10px color: white; text-align: center;">
            <h3>Inference(ms)</h3>
            <h2>{inference_speed_value:.2f}</h2>
        </div>
    """, unsafe_allow_html=True)


def update_histogram(results, class_selection):
    class_count = {name: 0 for name in class_selection}

    if not hasattr(results[0], 'obb') or results[0].obb is None:
        fig = px.bar(title="Waiting for Object Detection...", template="plotly_dark", height=325)
        fig.update_layout(
            title={'text': "Waiting for Object Detection...", 'y': 0.9, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'},
            xaxis={'visible': True, 'showgrid': True},
            yaxis={'visible': True, 'showgrid': True},
            annotations=[{'text': 'No Objects Detected', 'xref': 'paper', 'yref': 'paper', 'showarrow': False, 'font': {'size': 16}}]
        )
        return fig

    obb_results = results[0].obb

    for i in range(obb_results.shape[0]):
        if obb_results.data[i].size(0) >= 7:
            conf = obb_results.data[i][5].item()  # Confidence score
            cls_idx = int(obb_results.data[i][6].item())  # Class index
            if conf > 0.5:
                class_name = config.model.names[cls_idx]
                if class_name in class_selection:
                    class_count[class_name] += 1

    filtered_class_count = {k: v for k, v in class_count.items() if v > 0}
    if filtered_class_count:
        class_data = {"Classes": list(filtered_class_count.keys()), "Count": list(filtered_class_count.values())}
        df = pd.DataFrame(class_data)
        fig = px.bar(df, x="Classes", y="Count", title="Object Count", color="Classes", template="plotly_dark", height=325)
        fig.update_layout(
            title={'text': "Object Count", 'y': 0.9, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'},
            xaxis={'visible': True, 'showgrid': True},
            yaxis={'visible': True, 'showgrid': True}
        )
    else:
        fig = px.bar(title="Waiting for Object Detection...", template="plotly_dark", height=325)
        fig.update_layout(
            title={'text': "Waiting for Object Detection...", 'y': 0.9, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'},
            xaxis={'visible': True, 'showgrid': True},
            yaxis={'visible': True, 'showgrid': True},
            annotations=[{'text': 'No Objects Detected', 'xref': 'paper', 'yref': 'paper', 'showarrow': False, 'font': {'size': 16}}]
        )
    return fig


def update_object_proportion_chart(results, class_selection):
    vehicle_classes = class_selection
    vehicle_count = {cls: 0 for cls in vehicle_classes if cls in class_selection}

    if not hasattr(results[0], 'obb') or results[0].obb is None:
        fig = px.bar(title="Waiting for Object Detection...", template="plotly_dark", height=325)
        fig.update_layout(
            title={'text': "Waiting for Object Detection...", 'y': 0.9, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'},
            xaxis={'visible': True, 'showgrid': True},
            yaxis={'visible': True, 'showgrid': True}
        )
        return fig

    obb_results = results[0].obb

    for i in range(obb_results.shape[0]):
        if obb_results.data[i].size(0) >= 7:
            conf = obb_results.data[i][5].item()
            cls_idx = int(obb_results.data[i][6].item())
            if conf > 0.5:
                class_name = config.model.names[cls_idx]
                if class_name in vehicle_classes and class_name in class_selection:
                    vehicle_count[class_name] += 1

    filtered_vehicle_count = {k: v for k, v in vehicle_count.items() if v > 0}
    if filtered_vehicle_count:
        vehicle_data = {"Object Type": list(filtered_vehicle_count.keys()), "Count": list(filtered_vehicle_count.values())}
        df = pd.DataFrame(vehicle_data)
        fig = px.pie(df, names="Object Type", values="Count", title="Object Proportion", template="plotly_dark", height=500)
        fig.update_layout(
            title={'text': "Object Proportion", 'y': 0.9, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'},
            xaxis={'visible': True, 'showgrid': True},
            yaxis={'visible': True, 'showgrid': True}
        )
    else:
        fig = px.bar(title="Waiting for Object Detection...", template="plotly_dark", height=500)
        fig.update_layout(
            title={'text': "Waiting for Object Detection...", 'y': 0.9, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'},
            xaxis={'visible': True, 'showgrid': True},
            yaxis={'visible': True, 'showgrid': True},
            annotations=[{'text': 'No Objects Detected', 'xref': 'paper', 'yref': 'paper', 'showarrow': False, 'font': {'size': 16}}]
        )
    return fig


def calculate_area_proportions(results, frame_area, class_selection):
    area_dict = {name: 0 for name in class_selection}

    if not hasattr(results[0], 'obb') or results[0].obb is None:
        fig = px.bar(title="Waiting for Object Detection...", template="plotly_dark", height=325)
        fig.update_layout(
            title={'text': "Waiting for Object Detection...", 'y': 0.9, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'},
            xaxis={'visible': True, 'showgrid': True},
            yaxis={'visible': True, 'showgrid': True},
            annotations=[{'text': 'No Objects Detected', 'xref': 'paper', 'yref': 'paper', 'showarrow': False, 'font': {'size': 16}}]
        )
        return fig

    obb_results = results[0].obb

    for i in range(obb_results.shape[0]):
        if obb_results.data[i].size(0) >= 7:
            conf = obb_results.data[i][5].item()
            cls_idx = int(obb_results.data[i][6].item())
            if conf > 0.5:
                class_name = config.model.names[cls_idx]
                if class_name in class_selection:
                    width = obb_results.data[i][2].item()
                    height = obb_results.data[i][3].item()
                    area = width * height
                    area_dict[class_name] += area

    filtered_area_dict = {k: v for k, v in area_dict.items() if v > 0}
    if filtered_area_dict:
        area_data = {"Classes": list(filtered_area_dict.keys()), "Area": [area / frame_area * 100 for area in filtered_area_dict.values()]}
        df = pd.DataFrame(area_data)
        fig = px.pie(df, names="Classes", values="Area", title="Area Proportion", template="plotly_dark", height=325)
        fig.update_layout(
            title={'text': "Area Proportion", 'y': 0.9, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'},
            xaxis={'visible': True, 'showgrid': True},
            yaxis={'visible': True, 'showgrid': True}
        )
    else:
        fig = px.bar(title="Waiting for Object Detection...", template="plotly_dark", height=325)
        fig.update_layout(
            title={'text': "Waiting for Object Detection...", 'y': 0.9, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'},
            xaxis={'visible': True, 'showgrid': True},
            yaxis={'visible': True, 'showgrid': True},
            annotations=[{'text': 'No Objects Detected', 'xref': 'paper', 'yref': 'paper', 'showarrow': False, 'font': {'size': 16}}]
        )
    return fig


def update_traffic_graph(timestamp_data, class_selection):
    if not timestamp_data or not all('time' in entry and 'class' in entry and 'count' in entry for entry in timestamp_data):
        traffic_graph = px.line(title="No Traffic Data", template="plotly_dark", height=325)
        cumulative_graph = px.line(title="No Cumulative Data", template="plotly_dark", height=325)
        traffic_graph.update_layout(
            title={'text': "No Traffic Data", 'y': 0.9, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'},
            xaxis={'visible': True, 'showgrid': True},
            yaxis={'visible': True, 'showgrid': True}
        )
        cumulative_graph.update_layout(
            title={'text': "No Cumulative Data", 'y': 0.9, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'},
            xaxis={'visible': True, 'showgrid': True},
            yaxis={'visible': True, 'showgrid': True}
        )
        return traffic_graph, cumulative_graph

    df = pd.DataFrame(timestamp_data)
    df['time'] = pd.to_datetime(df['time'], format='%H:%M:%S')
    df = df.sort_values(by='time')
    df = df[df['class'].isin(class_selection)]
    
    df_second = df.groupby([df['time'].dt.floor('S'), 'class']).agg({'count': 'sum'}).reset_index()
    df_second.rename(columns={'count': 'count_per_second'}, inplace=True)
    df_second['cumulative_count'] = df_second.groupby('class')['count_per_second'].cumsum()
    df_second['time'] = df_second['time'].dt.strftime('%H:%M:%S')

    # Handle cases where there are multiple classes but only one timestamp
    if len(df_second['time'].unique()) == 1:
        unique_time = df_second['time'].iloc[0]
        for cls in df_second['class'].unique():
            single_class_data = df_second[df_second['class'] == cls]
            if len(single_class_data) == 1:
                dummy_row = single_class_data.iloc[0].copy()
                dummy_row['count_per_second'] = 0
                df_second = pd.concat([df_second, pd.DataFrame([dummy_row])], ignore_index=True)

    traffic_graph = px.line(df_second, x='time', y='count_per_second', color='class', title="Traffic Per Second by Class", template="plotly_dark", height=325)
    cumulative_graph = px.line(df_second, x='time', y='cumulative_count', color='class', title="Cumulative Traffic by Class", template="plotly_dark", height=325)

    traffic_graph.update_layout(
        title={'text': "Traffic Per Second by Class", 'y': 0.9, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'},
        xaxis={'visible': True, 'showgrid': True},
        yaxis={'visible': True, 'showgrid': True}
    )
    cumulative_graph.update_layout(
        title={'text': "Cumulative Traffic by Class", 'y': 0.9, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'},
        xaxis={'visible': True, 'showgrid': True},
        yaxis={'visible': True, 'showgrid': True}
    )

    return traffic_graph, cumulative_graph







# Function to generate a random color for each class
def get_random_color():
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

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



def apply_weather_effect(frame, weather_type="normal"):

    if weather_type == "Normal":
        # No augmentation, return the original frame
        return frame

    elif weather_type == "Rainy":
        # Apply a rainy effect
        rain_augmenter = iaa.Rain(drop_size=(0.10, 0.20), speed=(0.10, 0.30))
        rainy_frame = rain_augmenter(image=frame)
        return rainy_frame

    elif weather_type == "Cloudy":
        # Apply a cloudy effect
        cloud_augmenter = iaa.Clouds()
        cloudy_frame = cloud_augmenter(image=frame)
        return cloudy_frame

    elif weather_type == "Foggy":
        # Apply a foggy effect
        fog_augmenter = iaa.Fog()
        foggy_frame = fog_augmenter(image=frame)
        return foggy_frame

    else:
        # If the weather type is not recognized, return the original frame
        return frame


def generate_heatmap(frame, results, colormap=cv2.COLORMAP_JET):
    # Initialize a blank heatmap with the same dimensions as the frame
    heatmap = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.float32)

    # Check if there are any detections and process them
    if results is not None and results[0].obb is not None:
        obb_results = results[0].obb
        for i in range(obb_results.shape[0]):
            xywhr = obb_results.data[i][:5]  # Get bounding box (x_center, y_center, width, height, rotation)
            cls_idx = int(obb_results.data[i][6].item())  # Class index

            # Extract bounding box properties
            width = int(xywhr[2].item())
            height = int(xywhr[3].item())
            center_x = int(xywhr[0].item())
            center_y = int(xywhr[1].item())

            # Define the bounding box in the heatmap
            top_left = (max(0, center_x - width // 2), max(0, center_y - height // 2))
            bottom_right = (min(frame.shape[1], center_x + width // 2), min(frame.shape[0], center_y + height // 2))
            
            # Draw filled rectangle for detected object position
            cv2.rectangle(heatmap, top_left, bottom_right, (255,), thickness=-1)

    # Apply Gaussian blur to simulate gradient intensity around detected spots
    heatmap = cv2.GaussianBlur(heatmap, (31, 31), 0)

    # Normalize and apply colormap to create a visible heatmap effect
    heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
    colored_heatmap = cv2.applyColorMap(heatmap.astype(np.uint8), colormap)

    return colored_heatmap
