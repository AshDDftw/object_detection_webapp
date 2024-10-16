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


# class YOLOv8VideoProcessor:
#     def __init__(self):
#         self.confidence_threshold = 0.5
#         self.class_selection = []
#         self.results = None
#         self.current_class_count = {}
#         self.fps = 0
#         self.inference_speed = 0
#         self.frame_area = 0

#     def update_params(self, confidence_threshold, class_selection, weather_conditions):
#         self.confidence_threshold = confidence_threshold
#         self.class_selection = class_selection
#         self.weather_conditions = weather_conditions

#     def recv(self, frame):
#         img = frame.to_ndarray(format="bgr24")

#         # Apply any weather effect (if specified)
#         img = apply_weather_effect(img, self.weather_conditions)

#         # Perform object detection on the frame using the YOLOv8 model
#         results = config.model(img)

#         # Extract and update the results, class counts, fps, and inference speed
#         self.results = results
#         self.current_class_count = self.get_class_count(results)
#         self.frame_area = img.shape[0] * img.shape[1]

#         # Render and return the processed frame for display
#         return av.VideoFrame.from_ndarray(results.render()[0], format="bgr24")

#     def get_class_count(self, results):
#         # Initialize a dictionary for class counts
#         class_count = {name: 0 for name in config.model.names}

#         # Count objects detected in the frame
#         for result in results:
#             for box in result.boxes:
#                 cls = int(box.cls[0])  # Class index
#                 conf = float(box.conf[0])  # Confidence score
#                 if conf > self.confidence_threshold:
#                     class_name = config.model.names[cls]
#                     if class_name in self.class_selection:
#                         class_count[class_name] += 1

#         return class_count

# import cv2
# import av
# import threading
# import config
# from streamlit_webrtc import VideoProcessorBase

# class YOLOv8VideoProcessor(VideoProcessorBase):
#     def __init__(self):
#         self.confidence_threshold = 0.5  # Set a default confidence threshold
#         self.class_selection = []  # Selected classes
#         self.lock = threading.Lock()  # Ensure thread-safe access
#         self.results = None
#         self.current_class_count = {}
#         self.fps = 0
#         self.inference_speed = 0
#         self.frame_area = 0
#         self.model = config.model  # Load YOLOv8 model

#     def update_params(self, confidence_threshold, class_selection, weather_conditions='Normal'):
#         with self.lock:
#             self.confidence_threshold = confidence_threshold
#             self.class_selection = class_selection
#             self.weather_conditions = weather_conditions

#     def recv(self, frame):
#         img = frame.to_ndarray(format="bgr24")  # Get the video frame as a NumPy array

#         # Debugging: Check if the frame is captured
#         print("Captured frame shape:", img.shape)

#         # Process frame and collect metrics
#         try:
#             # Apply weather effect (e.g., cloudy) to the frame
#             processed_frame = apply_weather_effect(img, self.weather_conditions)

#             # Run YOLOv8 model on the processed frame
#             results = self.model(processed_frame)

#             # Measure inference speed and frame area
#             frame_area = processed_frame.shape[0] * processed_frame.shape[1]

#             # Extract and count detections
#             current_class_count = self.get_class_count(results)
#             fps = 1 / results.t if results.t > 0 else 0
#             inference_speed = results.t * 1000  # Convert inference time to ms

#         except Exception as e:
#             print("Error in frame processing:", str(e))
#             return av.VideoFrame.from_ndarray(processed_frame, format="bgr24")

#         # Update class state safely with a lock
#         with self.lock:
#             self.results = results
#             self.current_class_count = current_class_count
#             self.frame_area = frame_area
#             self.fps = fps
#             self.inference_speed = inference_speed

#         # Draw bounding boxes and labels on the frame
#         for result in results:
#             boxes = result.boxes
#             for box in boxes:
#                 cls = int(box.cls[0])  # Class index
#                 conf = float(box.conf[0])  # Confidence score
#                 if conf > self.confidence_threshold:
#                     class_name = self.model.names[cls]
#                     if class_name in self.class_selection:
#                         label = f'{class_name} {conf:.2f}'

#                         # Get the bounding box coordinates
#                         x1, y1, x2, y2 = map(int, box.xyxy[0])

#                         # Draw bounding box and label
#                         cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                         cv2.putText(processed_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

#         # Debugging: Ensure the image is properly processed
#         print("Processed frame ready for rendering")

#         # Return the processed frame
#         return av.VideoFrame.from_ndarray(processed_frame, format="bgr24")

#     def get_class_count(self, results):
#         # Initialize a dictionary for class counts
#         class_count = {name: 0 for name in self.model.names}

#         # Count objects detected in the frame
#         for result in results:
#             boxes = result.boxes
#             for box in boxes:
#                 cls = int(box.cls[0])
#                 conf = float(box.conf[0])
#                 if conf > self.confidence_threshold:
#                     class_name = self.model.names[cls]
#                     if class_name in self.class_selection:
#                         class_count[class_name] += 1

#         return class_count


# Function to update the metric blocks
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

# Function to calculate inference speed
def calculate_inference_speed(start_time, end_time):
    return (end_time - start_time) * 1000  # Convert to milliseconds

# Function to calculate FPS
def calculate_fps(start_time, end_time):
    return 1 / (end_time - start_time) if (end_time - start_time) > 0 else 0

# Function to update the histogram
def update_histogram(results, class_selection):
    class_count = {name: 0 for name in class_selection}

    # Ensure that results have the necessary attributes before proceeding
    if not hasattr(results[0], 'obb') or results[0].obb is None:
        # Handle the case where no detections were made (no obb data)
        fig = px.bar(title="Waiting for Object Detection...", template="plotly_dark", height=250)
        fig.update_layout(
    title={
        'text': "Waiting for Object Detection...",
        'y': 0.9,  # Vertical position (0 to 1)
        'x': 0.5,  # Horizontal position (0 to 1)
        'xanchor': 'center',
        'yanchor': 'top'
    },
    xaxis={'visible': False},
    yaxis={'visible': False},
    annotations=[{
        'text': 'No Objects Detected',
        'xref': 'paper',
        'yref': 'paper',
        'showarrow': False,
        'font': {'size': 16}
    }]
)
        return fig

    # Access OBB results
    obb_results = results[0].obb

    # Loop through each detected object and process the results
    for i in range(obb_results.shape[0]):
        # Check if the detection has at least 7 elements
        if obb_results.data[i].size(0) >= 7:
            conf = obb_results.data[i][5].item()  # Confidence score
            cls_idx = int(obb_results.data[i][6].item())  # Class index

            # Filter based on confidence threshold and selected classes
            if conf > 0.5:
                class_name = config.model.names[cls_idx]
                if class_name in class_selection:
                    class_count[class_name] += 1

    filtered_class_count = {k: v for k, v in class_count.items() if v > 0}

    if filtered_class_count:
        class_data = {
            "Classes": list(filtered_class_count.keys()),
            "Count": list(filtered_class_count.values())
        }
        df = pd.DataFrame(class_data)
        fig = px.bar(df, x="Classes", y="Count", title="Object Count",
                     color="Classes", template="plotly_dark", height=250)
        fig.update_layout(
    title={
        'text': "Object Count",
        'y': 0.9,  # Vertical position (0 to 1)
        'x': 0.5,  # Horizontal position (0 to 1)
        'xanchor': 'center',
        'yanchor': 'top'
    },
    xaxis={'visible': False},
    yaxis={'visible': False},
)
    else:
        fig = px.bar(title="Waiting for Object Detection...", template="plotly_dark", height=250)
        fig.update_layout(
    title={
        'text': "Waiting for Object Detection...",
        'y': 0.9,  # Vertical position (0 to 1)
        'x': 0.5,  # Horizontal position (0 to 1)
        'xanchor': 'center',
        'yanchor': 'top'
    },
    xaxis={'visible': False},
    yaxis={'visible': False},
    annotations=[{
        'text': 'No Objects Detected',
        'xref': 'paper',
        'yref': 'paper',
        'showarrow': False,
        'font': {'size': 16}
    }]
)
    return fig


# Function to update the vehicle proportion pie chart
def update_vehicle_proportion_chart(results, class_selection):
    vehicle_classes = class_selection # Adjust vehicle classes as needed
    vehicle_count = {cls: 0 for cls in vehicle_classes if cls in class_selection}

    
    if not hasattr(results[0], 'obb') or results[0].obb is None:
        # Handle the case where no detections were made (no obb data)
        fig = px.bar(title="Waiting for Object Detection...", template="plotly_dark", height=250)
        fig.update_layout(xaxis={'visible': False}, yaxis={'visible': False}, annotations=[{
            'text': 'No Objects Detected',
            'xref': 'paper',
            'yref': 'paper',
            'showarrow': False,
            'font': {'size': 16}
        }])
        return fig
    
    # Access OBB results
    obb_results = results[0].obb

    # Loop through each detected object and process the results
    for i in range(obb_results.shape[0]):
        # Check if the detection has at least 7 elements
        if obb_results.data[i].size(0) >= 7:
            # Extract confidence and class info
            conf = obb_results.data[i][5].item()  # Confidence score
            cls_idx = int(obb_results.data[i][6].item())  # Class index
            
            # Filter based on confidence threshold
            if conf > 0.5:
                class_name = config.model.names[cls_idx]
                if class_name in vehicle_classes and class_name in class_selection:
                    vehicle_count[class_name] += 1

    # Filter out classes with zero counts
    filtered_vehicle_count = {k: v for k, v in vehicle_count.items() if v > 0}

    if filtered_vehicle_count:
        vehicle_data = {"Object Type": list(filtered_vehicle_count.keys()), "Count": list(filtered_vehicle_count.values())}
        df = pd.DataFrame(vehicle_data)
        print('vvv',df)
        fig = px.pie(df, names="Object Type", values="Count", title="Object Proportion",
                     template="plotly_dark", color_discrete_sequence=px.colors.sequential.RdBu, height=325)
        fig.update_layout(
    title={
        'text': "Object Proportion",
        'y': 0.9,  # Vertical position (0 to 1)
        'x': 0.5,  # Horizontal position (0 to 1)
        'xanchor': 'center',
        'yanchor': 'top'
    },
    xaxis={'visible': False},
    yaxis={'visible': False},
)
    else:
        fig = px.bar(title="Waiting for Object Detection...", template="plotly_dark", height=250)
        fig.update_layout(
    title={
        'text': "Waiting for Object Detection...",
        'y': 0.9,  # Vertical position (0 to 1)
        'x': 0.5,  # Horizontal position (0 to 1)
        'xanchor': 'center',
        'yanchor': 'top'
    },
    xaxis={'visible': False},
    yaxis={'visible': False},
    annotations=[{
        'text': 'No Objects Detected',
        'xref': 'paper',
        'yref': 'paper',
        'showarrow': False,
        'font': {'size': 16}
    }]
)

    return fig


# Function to calculate area proportions and plot a pie chart
def calculate_area_proportions(results, frame_area, class_selection):
    area_dict = {name: 0 for name in class_selection}
    
    if not hasattr(results[0], 'obb') or results[0].obb is None:
        # Handle the case where no detections were made (no obb data)
        fig = px.bar(title="Waiting for Object Detection...", template="plotly_dark", height=250)
        fig.update_layout(xaxis={'visible': False}, yaxis={'visible': False}, annotations=[{
            'text': 'No Objects Detected',
            'xref': 'paper',
            'yref': 'paper',
            'showarrow': False,
            'font': {'size': 16}
        }])
        return fig
    # Access OBB results
    obb_results = results[0].obb

    # Loop through each detected object and process the results
    for i in range(obb_results.shape[0]):
        # Check if the detection has the minimum 7 attributes (x_center, y_center, width, height, rotation, confidence, class_id)
        if obb_results.data[i].size(0) >= 7:
            # Extract the bounding box coordinates and class info
            conf = obb_results.data[i][5].item()  # Get the confidence score
            cls_idx = int(obb_results.data[i][6].item())  # Get the class index
            
            # Filter results based on confidence threshold and selected classes
            if conf > 0.5:  # Use confidence threshold
                class_name = config.model.names[cls_idx]
                if class_name in class_selection:
                    # Calculate the area of the detected bounding box
                    width = obb_results.data[i][2].item()  # Width of the bounding box
                    height = obb_results.data[i][3].item()  # Height of the bounding box
                    area = width * height
                    area_dict[class_name] += area

    # Filter out zero-area classes
    filtered_area_dict = {k: v for k, v in area_dict.items() if v > 0}

    if filtered_area_dict:
        area_data = {
            "Classes": list(filtered_area_dict.keys()),
            "Area": [area / frame_area * 100 for area in filtered_area_dict.values()]
        }
        df = pd.DataFrame(area_data)
        print('ccc',df)
        fig = px.pie(df, names="Classes", values="Area", title="Area Proportion",
                     template="plotly_dark", color_discrete_sequence=px.colors.sequential.RdBu, height=280)
        fig.update_layout(
    title={
        'text': "Area Proportion",
        'y': 0.9,  # Vertical position (0 to 1)
        'x': 0.5,  # Horizontal position (0 to 1)
        'xanchor': 'center',
        'yanchor': 'top'
    },
    xaxis={'visible': False},
    yaxis={'visible': False},
)
    else:
        fig = px.bar(title="Waiting for Object Detection...", template="plotly_dark", height=250)
        fig.update_layout(
    title={
        'text': "Waiting for Object Detection...",
        'y': 0.9,  # Vertical position (0 to 1)
        'x': 0.5,  # Horizontal position (0 to 1)
        'xanchor': 'center',
        'yanchor': 'top'
    },
    xaxis={'visible': False},
    yaxis={'visible': False},
    annotations=[{
        'text': 'No Objects Detected',
        'xref': 'paper',
        'yref': 'paper',
        'showarrow': False,
        'font': {'size': 16}
    }]
)

    return fig



def update_traffic_graph(timestamp_data, class_selection):
    # Check if timestamp_data is valid (non-empty list and contains the necessary keys)
    if not timestamp_data or not all('time' in entry and 'class' in entry and 'count' in entry for entry in timestamp_data):
        print('incoming', timestamp_data)
        # Return empty figures if no valid data exists
        traffic_graph = px.line(title="No Traffic Data", template="plotly_dark", height=250)
        cumulative_graph = px.line(title="No Cumulative Data", template="plotly_dark", height=325)
        traffic_graph.update_layout(xaxis={'visible': True, 'showgrid': True}, yaxis={'visible': True, 'showgrid': True}, title_x=0.5)
        cumulative_graph.update_layout(xaxis={'visible': True, 'showgrid': True}, yaxis={'visible': True, 'showgrid': True}, title_x=0.5)
        return traffic_graph, cumulative_graph

    # Convert the list of dictionaries to a DataFrame
    df = pd.DataFrame(timestamp_data)

    # Ensure that the 'time' column is present
    if 'time' not in df.columns or df.empty:
        traffic_graph = px.line(title="No Traffic Data", template="plotly_dark", height=250)
        cumulative_graph = px.line(title="No Cumulative Data", template="plotly_dark", height=325)
        traffic_graph.update_layout(xaxis={'visible': True, 'showgrid': True}, yaxis={'visible': True, 'showgrid': True})
        cumulative_graph.update_layout(xaxis={'visible': True, 'showgrid': True}, yaxis={'visible': True, 'showgrid': True})
        return traffic_graph, cumulative_graph

    # Filter by class selection
    df = df[df['class'].isin(class_selection)]
    
    # Convert 'time' column to datetime format for proper grouping
    df['time'] = pd.to_datetime(df['time'], format='%H:%M:%S')
    df = df.sort_values(by='time')  # Ensure data is sorted by time

    # Group data per second and by class
    df_second = df.groupby([df['time'].dt.floor('S'), 'class']).agg({'count': 'sum'}).reset_index()
    df_second.rename(columns={'count': 'count_per_second'}, inplace=True)

    # Calculate cumulative count per class
    df_second['cumulative_count'] = df_second.groupby('class')['count_per_second'].cumsum()

    # Convert time back to HH:MM:SS format for plotting
    df_second['time'] = df_second['time'].dt.strftime('%H:%M:%S')

    # Handle case for single data point
    if len(df_second) == 1:
        df_second = df_second.append({'time': df_second['time'].iloc[0], 'class': df_second['class'].iloc[0], 'count_per_second': 0, 'cumulative_count': 0}, ignore_index=True)

    # Traffic per second for each class
    traffic_graph = px.line(df_second, x='time', y='count_per_second', color='class', title="Traffic Per Second by Class", 
                            template="plotly_dark", height=250)

    # Cumulative traffic over time for each class
    cumulative_graph = px.line(df_second, x='time', y='cumulative_count', color='class', title="Cumulative Traffic by Class", 
                               template="plotly_dark", height=325)

    # Ensure axes visibility and grid lines, and center the title
    traffic_graph.update_layout(xaxis={'visible': True, 'showgrid': True}, yaxis={'visible': True, 'showgrid': True})
    cumulative_graph.update_layout(xaxis={'visible': True, 'showgrid': True}, yaxis={'visible': True, 'showgrid': True})

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
    # Create a blank heatmap with the same dimensions as the frame
    heatmap = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.float32)

    # Check if there are any detections and process them
    if results is not None and results[0].obb is not None:
        obb_results = results[0].obb
        for i in range(obb_results.shape[0]):
            xywhr = obb_results.data[i][:5]  # Get the bounding box (x_center, y_center, width, height, rotation)
            cls_idx = int(obb_results.data[i][6].item())  # Get the class index

            # Draw a rectangle on the heatmap where the object was detected
            width = int(xywhr[2].item())  # Width of the bounding box
            height = int(xywhr[3].item())  # Height of the bounding box
            center_x = int(xywhr[0].item())  # X-center of the bounding box
            center_y = int(xywhr[1].item())  # Y-center of the bounding box

            # Create the bounding box as a rectangle in the heatmap (use width and height)
            top_left = (max(0, center_x - width // 2), max(0, center_y - height // 2))
            bottom_right = (min(frame.shape[1], center_x + width // 2), min(frame.shape[0], center_y + height // 2))
            cv2.rectangle(heatmap, top_left, bottom_right, (255,), thickness=-1)

    # Normalize the heatmap to range between 0 and 255
    heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)

    # Apply colormap to the heatmap
    colored_heatmap = cv2.applyColorMap(heatmap.astype(np.uint8), colormap)

    # Overlay the heatmap on the original frame
    overlaid_frame = cv2.addWeighted(frame, 0.6, colored_heatmap, 0.4, 0)

    return overlaid_frame
