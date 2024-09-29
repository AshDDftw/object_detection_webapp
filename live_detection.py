import time
import threading
import av
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
from common import (
    model,
    update_histogram,
    calculate_area_proportions,
    update_vehicle_proportion_chart,
    update_traffic_graph,
    update_timestamp_line_graph,
    process_frame,
    confidence_threshold,
    class_selection,
    timestamp_data,
    total_objects,
    fps_display,
    resolution_display,
    inference_speed_display,
)

# YOLOv5 video processor class for WebRTC live streaming
class YOLOv5VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.confidence_threshold = confidence_threshold  # Confidence threshold slider value from the sidebar
        self.class_selection = class_selection  # Class selection from the sidebar
        self.lock = threading.Lock()  # Ensure thread-safe access by adding a lock
        self.results = None
        self.current_class_count = {}
        self.frame_area = 0
        self.fps = 0
        self.inference_speed = 0

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")  # Get the video frame as a NumPy array

        # Process frame and collect metrics
        results, current_class_count, frame_area, fps, inference_speed = process_frame(img)

        # Update class state safely with a lock
        with self.lock:
            self.results = results
            self.current_class_count = current_class_count
            self.frame_area = frame_area
            self.fps = fps
            self.inference_speed = inference_speed

        return av.VideoFrame.from_ndarray(img, format="bgr24")


def run_live_detection():
    frame_placeholder = st.empty()
    histogram_placeholder = st.empty()
    pie_chart_placeholder = st.empty()
    vehicle_pie_chart_placeholder = st.empty()
    traffic_graph_placeholder = st.empty()
    cumulative_graph_placeholder = st.empty()
    timestamp_histogram_placeholder = st.empty()

    ctx = webrtc_streamer(
        key="object-detection",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=YOLOv5VideoProcessor,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    # Loop for continuously updating metrics while webcam is running
    while ctx.state.playing:
        if ctx.video_processor:
            processor = ctx.video_processor
            if hasattr(processor, "lock"):
                with processor.lock:
                    if processor.results:
                        # Update metrics and charts
                        total = sum(processor.current_class_count.values())
                        total_objects.metric("Total Objects", total)
                        fps_display.metric("FPS", f"{processor.fps:.2f}")
                        resolution_display.metric("Resolution", f"{640}x{480}")
                        inference_speed_display.metric(
                            "Inference Speed (ms)", f"{processor.inference_speed:.2f}"
                        )

                        histogram = update_histogram(processor.current_class_count)
                        pie_chart = calculate_area_proportions(
                            processor.results, processor.frame_area
                        )
                        vehicle_pie_chart = update_vehicle_proportion_chart(
                            processor.current_class_count
                        )

                        histogram_placeholder.plotly_chart(histogram, use_container_width=True)
                        pie_chart_placeholder.plotly_chart(pie_chart, use_container_width=True)
                        vehicle_pie_chart_placeholder.plotly_chart(vehicle_pie_chart, use_container_width=True)

                        traffic_graph, cumulative_graph = update_traffic_graph(timestamp_data)
                        traffic_graph_placeholder.plotly_chart(traffic_graph, use_container_width=True)
                        cumulative_graph_placeholder.plotly_chart(cumulative_graph, use_container_width=True)

                        line_graph = update_timestamp_line_graph(timestamp_data)
                        timestamp_histogram_placeholder.plotly_chart(line_graph, use_container_width=True)

        time.sleep(0.1)
