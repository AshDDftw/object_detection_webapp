import streamlit as st
from live_detection import run_live_detection
from image_detection import run_image_detection
from video_detection import run_video_detection
import config

# Set up Streamlit
st.set_page_config(page_title="Real-Time YOLOv5 Object Detection", page_icon="✅", layout="wide")

# Custom CSS for styling
st.markdown("""
    <style>
    .css-18e3th9 { padding-top: 0.5rem; padding-bottom: 0.5rem; }
    .css-1d391kg { padding-top: 0.5rem; padding-bottom: 0.5rem; }
    .stApp {
    background: linear-gradient(135deg, #191970, #3a8ddb);
    padding-left: 20px;
    padding-right: 20px;
}


    /* Style the title */
    .title {
        font-family: 'Roboto', sans-serif;
        color: #FBE9D0;
        font-size: 3rem;
        font-weight: 700;
        text-align: center;
        margin-top: 20px;
        margin-bottom: 20px;
    }

    /* Professional colors for blocks */
    .block { 
        background-color: #2E3B4E; 
        padding: 10px; 
        border-radius: 10px; 
        color: white; 
        text-align: center; 
        margin-bottom: 20px; /* Leave space between elements */
    }
    .block-orange { 
        background-color: #F39C12; 
        padding: 10px; 
        border-radius: 10px; 
        color: white; 
        text-align: center; 
        margin-bottom: 20px; /* Leave space between elements */
    }
    .block-yellow { 
        background-color: #F1C40F; 
        padding: 10px; 
        border-radius: 10px; 
        color: black; 
        text-align: center; 
        margin-bottom: 20px; /* Leave space between elements */
    }
    .block-green { 
        background-color: #27AE60; 
        padding: 10px; 
        border-radius: 10px; 
        color: white; 
        text-align: center; 
        margin-bottom: 20px; /* Leave space between elements */
    }

    /* Add padding between columns and sections */
    .stColumn { padding-right: 10px; padding-left: 10px; } /* Adds space between columns */
    .stFrame { margin-top: 20px; } /* Adds space between frame and other elements */

    /* Add margin below the metric blocks */
    .metric-block { margin-bottom: 30px !important; }  /* Add space below metric blocks */

    /* Add spacing between sections */
    .stBlock-container { margin-bottom: 20px; }
    </style>
""", unsafe_allow_html=True)

# Add title with new styling
st.markdown('<h1 class="title">Aerial Vision: Object Detection Dashboard</h1>', unsafe_allow_html=True)

# Metric placeholders at the top
metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4, gap="small")
with metric_col1:
    total_objects_placeholder = st.empty()
with metric_col2:
    fps_placeholder = st.empty()
with metric_col3:
    resolution_placeholder = st.empty()
with metric_col4:
    inference_speed_placeholder = st.empty()

st.markdown("""
    <style>
        .metric-block {
            margin-bottom: 20px;
        }

        /* Round borders for placeholder elements */
        .metric-block, .stEmpty {
            border-radius: 15px;
            border: 2px solid #ccc;  /* Light border for the elements */
            padding: 10px;
        }

        /* Optional: Add margin below the metric block */
        .stEmpty {
            margin-bottom: 20px;
        }
    </style>
""", unsafe_allow_html=True)



# Main layout: Vehicle Proportion and Traffic Graph on the Left, Input Frame on the Right
left_col, right_col = st.columns([2, 3], gap="medium")

with left_col:
    lc , rc = st.columns(2, gap="medium")
    with lc:
        pie_chart_placeholder_1 = st.empty()  # Vehicle Proportion
    with rc:
        pie_chart_placeholder_2 = st.empty()  # Object proportion pie chart
    traffic_graph_placeholder = st.empty()  # Traffic Graph
    class_distribution_placeholder = st.empty()  # Below the frame: Class distribution histogram (full width)
    cumulative_graph_placeholder = st.empty()  # Heatmap placeholder

with right_col:
    frame_placeholder = st.empty()  # Display input image/video
    cumulative_graph_placeholder = st.empty()  # Heatmap placeholder


st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)  # Add space before the frame


# Full-width Class Distribution Histogram Below the Frame and Graphs
st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)  # Add space between sections
st.markdown('<hr style="border:1px solid #FFFFFF;">', unsafe_allow_html=True)  # Horizontal line between sections

# # Bottom section: Object Proportion Pie Chart and Heatmap side by side
bottom_col1, bottom_col2 = st.columns(2, gap="medium")
# with bottom_col1:
    


# Sidebar options for detection mode
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

# Function to update metric blocks dynamically
def update_metric_blocks(total_objects, fps_value, resolution_width, resolution_height, inference_speed_value):
    total_objects_placeholder.markdown(f"""
        <div style=" background-color: #ed2d37; 
        padding: 10px; 
        border-radius: 10px; 
        color: white; 
        text-align: center;">
            <h3>Total Objects</h3>
            <h2>{total_objects}</h2>
        </div>
    """, unsafe_allow_html=True)

    fps_placeholder.markdown(f"""
        <div style="background-color: #F39C12; 
        padding: 10px; 
        border-radius: 10px; 
        color: white; 
        text-align: center;">
            <h3>FPS</h3>
            <h2>{fps_value:.2f}</h2>
        </div>
    """, unsafe_allow_html=True)

    resolution_placeholder.markdown(f"""
        <div style="background-color: #F1C40F; 
        padding: 10px; 
        border-radius: 10px; 
        color: black; 
        text-align: center;">
            <h3>Resolution</h3>
            <h2>{resolution_width}x{resolution_height}</h2>
        </div>
    """, unsafe_allow_html=True)

    inference_speed_placeholder.markdown(f"""
        <div style="background-color: #27AE60; 
        padding: 10px; 
        border-radius: 10px; 
        color: black; 
        text-align: center;">
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
