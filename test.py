import streamlit as st
from live_detection import run_live_detection
from image_detection import run_image_detection
from video_detection import run_video_detection
import config

# Set up Streamlit
st.set_page_config(page_title="Real-Time YOLOv5 Object Detection", page_icon="✅", layout="wide")

st.markdown("""
    <style>
    .css-18e3th9 { padding-top: 0.5rem; padding-bottom: 0.5rem; }
    .css-1d391kg { padding-top: 0.5rem; padding-bottom: 0.5rem; }
    .stApp { background-color: #6b38b5; padding-left: 0px; padding-right: 0px;} /* Light background */

    /* Style the title */
    .title {
        font-family: 'Roboto', sans-serif;
        color: #FBE9D0;
        font-size: 3rem;
        font-weight: 700;
        text-align: center;
        margin-top: 20px;
        margin-bottom: 20px;
        background-color:#FBE9D0:
    }

    /* Make the content edge-to-edge */
    .main {
        padding-left: 0px;
        padding-right: 0px;
    }

    /* Professional colors for blocks */
    .block { 
        background-color: #2E3B4E; 
        padding: 10px; 
        border-radius: 10px; 
        color: white; 
        text-align: center; 
        margin-bottom: 30px; /* Increase margin-bottom for more padding */
    }
    .block-orange { 
        background-color: #F39C12; 
        padding: 10px; 
        border-radius: 10px; 
        color: white; 
        text-align: center; 
        margin-bottom: 30px; /* Increase margin-bottom for more padding */
    }
    .block-yellow { 
        background-color: #F1C40F; 
        padding: 10px; 
        border-radius: 10px; 
        color: black; 
        text-align: center; 
        margin-bottom: 30px; /* Increase margin-bottom for more padding */
    }
    .block-green { 
        background-color: #27AE60; 
        padding: 10px; 
        border-radius: 10px; 
        color: white; 
        text-align: center; 
        margin-bottom: 30px; /* Increase margin-bottom for more padding */
    }

    /* Remove padding and margin at the bottom to prevent large gap */
    .main .block-container {
        padding-bottom: 0px !important;
        margin-bottom: 0px !important;
    }

    /* Add margin-top to the frame to bring it down */
    .stFrame {
        margin-top: 30px; /* Adjust this value as needed */
    }
    </style>
    """, unsafe_allow_html=True)

# Add title with new styling
st.markdown('<h1 class="title">Object Detection Dashboard</h1>', unsafe_allow_html=True)


# Sidebar options
detection_mode = st.sidebar.radio("Detection Mode", ("Live Detection", "Upload Image", "Upload Video"))




class_selection = st.sidebar.multiselect("Select Classes to Detect", options=list(config.model.names.values()), default=None)
if not class_selection:
    class_selection = list(config.model.names.values())

confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, config.confidence_threshold, 0.05)

# Add weather dropdown to the sidebar
weather_conditions = st.sidebar.selectbox(
    "Select Weather Condition",
    ("Normal", "Rainy", "Cloudy", "Foggy", "Other")
)

# Metric placeholders above the frame with padding
metric_col1, metric_col2, metric_col3, metric_col4 = st.columns([1, 1, 1, 1])
with metric_col1:
    total_objects_placeholder = st.empty()
with metric_col2:
    fps_placeholder = st.empty()
with metric_col3:
    resolution_placeholder = st.empty()
with metric_col4:
    inference_speed_placeholder = st.empty()

# Layout for the frame, pie charts, and traffic plots with padding between them
main_col1, main_col2, main_col3 = st.columns([2, 4, 2], gap="medium")

with main_col1:
    pie_chart_placeholder_1 = st.empty()  # Left pie chart placeholder
    pie_chart_placeholder_2 = st.empty()  # Left second pie chart placeholder

with main_col2:
    frame_placeholder = st.empty()  # Center frame placeholder
    class_distribution_placeholder = st.empty()

with main_col3:
    traffic_graph_placeholder = st.empty()  # Right traffic graph placeholder
    cumulative_graph_placeholder = st.empty()  # Right cumulative traffic graph placeholder



# Function to update metric blocks dynamically
def update_metric_blocks(total_objects, fps_value, resolution_width, resolution_height, inference_speed_value):
    total_objects_placeholder.markdown(f"""
        <div style=" background-color: #ed2d37; 
        padding: 10px; 
        border-radius: 10px; 
        color: white; 
        text-align: center; 
        margin-bottom: 30px;">
            <h3>Total Objects</h3>
            <h2>{total_objects}</h2>
        </div>
    """, unsafe_allow_html=True)

    fps_placeholder.markdown(f"""
        <div style="background-color: #F39C12; 
        padding: 10px; 
        border-radius: 10px; 
        color: white; 
        text-align: center; 
        margin-bottom: 30px; ">
            <h3>FPS</h3>
            <h2>{fps_value:.2f}</h2>
        </div>
    """, unsafe_allow_html=True)

    resolution_placeholder.markdown(f"""
        <div style="background-color: #F1C40F; 
        padding: 10px; 
        border-radius: 10px; 
        color: black; 
        text-align: center; 
        margin-bottom: 30px;">
            <h3>Resolution</h3>
            <h2>{resolution_width}x{resolution_height}</h2>
        </div>
    """, unsafe_allow_html=True)

    inference_speed_placeholder.markdown(f"""
        <div style="background-color: #27AE60; 
        padding: 10px; 
        border-radius: 10px; 
        color: black; 
        text-align: center; 
        margin-bottom: 30px;">
            <h3>Inference (ms)</h3>
            <h2>{inference_speed_value:.2f}</h2>
        </div>
    """, unsafe_allow_html=True)


# Switch between detection modes
if detection_mode == "Live Detection":
    run_live_detection(
        confidence_threshold,  # Pass the dynamically selected confidence threshold
        class_selection,  # Pass the dynamically selected class selection
        frame_placeholder,
        pie_chart_placeholder_1, 
        pie_chart_placeholder_2, 
        traffic_graph_placeholder, 
        cumulative_graph_placeholder,
        class_distribution_placeholder,
        update_metric_blocks,
        weather_conditions
    )

elif detection_mode == "Upload Image":
    run_image_detection(
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
    )

elif detection_mode == "Upload Video":
    run_video_detection(
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
    )
