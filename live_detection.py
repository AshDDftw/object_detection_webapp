import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import config
from app_utils import YOLOv5VideoProcessor, update_histogram, calculate_area_proportions, update_vehicle_proportion_chart, update_traffic_graph
from datetime import datetime, timedelta
import time
import cv2

# Global variable to store traffic data (last 10 seconds)
traffic_data = []

def run_live_detection(
    confidence_threshold,  # Add confidence threshold parameter
    class_selection,  # Add class selection parameter
    frame_placeholder, 
    pie_chart_placeholder_1, 
    pie_chart_placeholder_2, 
    traffic_graph_placeholder, 
    cumulative_graph_placeholder,
    class_distribution_placeholder,
    update_metric_blocks  # Function to update metric blocks dynamically
):
    # Create a WebRTC streamer instance for live detection
    ctx = webrtc_streamer(
        key="object-detection",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=YOLOv5VideoProcessor,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    # Loop for continuously updating metrics while the webcam is running
    while ctx.state.playing:
        if ctx.video_processor:
            processor = ctx.video_processor
            # print(processor)
            # Dynamically update the confidence threshold and class selection
            processor.update_params(confidence_threshold, class_selection)

            if hasattr(processor, 'lock'):
                with processor.lock:
                    if processor.results:
                        print(processor.results)
                        # Debugging: Display class counts
                        st.write("Class counts: ", processor.current_class_count)

                        # Get the processed frame and metrics
                        total_objects = sum(processor.current_class_count.values())
                        print(total_objects)
                        fps_value = processor.fps
                        processed_frame = processor.results.render()[0]  # Get the processed frame
                        resolution_height, resolution_width = processed_frame.shape[:2]
                        inference_speed_value = processor.inference_speed

                        # Update the metric blocks
                        update_metric_blocks(
                            total_objects, fps_value, resolution_width, resolution_height, inference_speed_value
                        )

                        # Convert the processed frame to RGB and display it
                        rgb_frame = cv2.cvtColor(processor.results.render()[0], cv2.COLOR_BGR2RGB)
                        frame_placeholder.image(rgb_frame, use_column_width=True)

                        # Update charts below the video frame
                        histogram = update_histogram(processor.current_class_count)
                        pie_chart_1 = calculate_area_proportions(processor.results, processor.frame_area, class_selection)
                        pie_chart_2 = update_vehicle_proportion_chart(processor.current_class_count)

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
                        # print(traffic_data)

                        # Pass the updated traffic data to update the traffic and cumulative graphs
                        traffic_graph, cumulative_graph = update_traffic_graph(traffic_data)
                        traffic_graph_placeholder.plotly_chart(traffic_graph, use_container_width=True)
                        cumulative_graph_placeholder.plotly_chart(cumulative_graph, use_container_width=True)

                        # Update class distribution histogram
                        class_distribution_placeholder.plotly_chart(histogram, use_container_width=True)

        time.sleep(0.1)  # Small sleep to prevent resource overconsumption
