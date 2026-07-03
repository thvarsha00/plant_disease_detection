import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import tempfile
import os
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# ----------------------------------------------------------------------------
# Page Configuration
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Plant Disease Recognition System",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------------------------------------------------------
# Custom Styling
# ----------------------------------------------------------------------------
st.markdown("""
    <style>
        .main {
            background-color: #0e1117;
        }
        h1, h2, h3 {
            font-weight: 600;
            letter-spacing: -0.5px;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 24px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 44px;
            padding: 0 8px;
            font-weight: 500;
        }
        .result-card {
            background-color: #1a1d24;
            border: 1px solid #2d313c;
            border-radius: 10px;
            padding: 20px 24px;
            margin-top: 12px;
        }
        .result-label {
            font-size: 13px;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: #8a8f98;
            margin-bottom: 4px;
        }
        .result-value {
            font-size: 22px;
            font-weight: 600;
            color: #f0f2f6;
        }
        .confidence-value {
            font-size: 16px;
            font-weight: 500;
            color: #4ade80;
        }
        .section-divider {
            border-top: 1px solid #2d313c;
            margin: 24px 0;
        }
        footer {visibility: hidden;}
        #MainMenu {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
MODEL_PATH = "trained_model.h5"
LAST_CONV_LAYER_NAME = "conv2d_9"
IMAGE_SIZE = (128, 128)

CLASS_NAMES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot', 'Tomato___Tomato_mosaic_virus', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___healthy'
]


# ----------------------------------------------------------------------------
# Model Loading
# ----------------------------------------------------------------------------
@st.cache_resource()
def load_model():
    """Loads and caches the trained Keras model."""
    return tf.keras.models.load_model(MODEL_PATH)


# ----------------------------------------------------------------------------
# Core Logic
# ----------------------------------------------------------------------------
def make_cam_heatmap(model, img_array, last_conv_layer_name):
    """Generates a Class Activation Map heatmap for the given input."""
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]

    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)

    return heatmap.numpy()


def model_prediction(model, image):
    """Runs inference on a PIL image and returns the predicted class index and confidence."""
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])
    prediction = model.predict(input_arr, verbose=0)
    result_index = np.argmax(prediction)
    return result_index, prediction[0][result_index]


def overlay_heatmap(image_path, heatmap, alpha=0.4, size=(256, 256)):
    """Overlays a CAM heatmap on the original image."""
    original_image = cv2.imread(image_path)
    original_image = cv2.resize(original_image, size)
    heatmap_resized = cv2.resize(heatmap, size)
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    return cv2.addWeighted(original_image, 1 - alpha, heatmap_colored, alpha, 0)


def format_class_name(raw_name):
    """Converts a raw class label into a readable display string."""
    plant, _, condition = raw_name.partition("___")
    plant = plant.replace("_", " ").strip()
    condition = condition.replace("_", " ").strip()
    return plant, condition if condition else "Healthy"


# ----------------------------------------------------------------------------
# Sidebar Navigation
# ----------------------------------------------------------------------------
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio(
    "Select a page",
    ["Home", "About", "Disease Recognition"],
    label_visibility="collapsed"
)

st.sidebar.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
st.sidebar.markdown("**Model**")
st.sidebar.caption("Convolutional Neural Network")
st.sidebar.markdown("**Classes**")
st.sidebar.caption(f"{len(CLASS_NAMES)} plant disease categories")


# ----------------------------------------------------------------------------
# Home Page
# ----------------------------------------------------------------------------
if app_mode == "Home":
    st.title("Plant Disease Recognition System")
    st.caption("Deep learning-based leaf disease identification for crop health monitoring")

    if os.path.exists("home_page.jpeg"):
        st.image("home_page.jpeg", width="stretch")
    else:
        st.info("Home page image not found. Place 'home_page.jpeg' in the project directory to display it here.")

    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Upload & Detect**")
        st.caption("Upload a leaf image and get an instant diagnosis with a confidence score.")
    with col2:
        st.markdown("**Model Explainability**")
        st.caption("View a Class Activation Map showing which regions influenced the prediction.")
    with col3:
        st.markdown("**Live Camera Mode**")
        st.caption("Run real-time detection directly from a webcam feed.")

    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    st.markdown(
        "Navigate to **Disease Recognition** from the sidebar to begin, "
        "or visit **About** to learn more about the underlying model."
    )


# ----------------------------------------------------------------------------
# About Page
# ----------------------------------------------------------------------------
elif app_mode == "About":
    st.title("About This Project")

    st.markdown("### Overview")
    st.write(
        "This system uses a Convolutional Neural Network trained to identify plant diseases "
        "from leaf images. It supports 38 distinct classes spanning multiple crop species and "
        "provides both a prediction and a visual explanation of the model's decision."
    )

    st.markdown("### Model Details")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='result-card'>", unsafe_allow_html=True)
        st.markdown("<div class='result-label'>Architecture</div>", unsafe_allow_html=True)
        st.markdown("<div class='result-value'>CNN (TensorFlow / Keras)</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='result-card'>", unsafe_allow_html=True)
        st.markdown("<div class='result-label'>Output Classes</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='result-value'>{len(CLASS_NAMES)}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

    st.markdown("### Explainability")
    st.write(
        "Predictions are accompanied by a Class Activation Map (CAM), generated from the final "
        "convolutional layer, to highlight the image regions that most strongly influenced the "
        "model's decision."
    )

    st.markdown("### Disclaimer")
    st.warning(
        "This tool is intended for educational and informational purposes only. "
        "For professional agricultural guidance, consult a certified expert."
    )


# ----------------------------------------------------------------------------
# Disease Recognition Page
# ----------------------------------------------------------------------------
elif app_mode == "Disease Recognition":
    st.title("Disease Recognition")
    st.caption("Upload a leaf image or use your camera to run a live diagnosis")

    model = load_model()

    tab1, tab2 = st.tabs(["Upload Image", "Live Camera"])

    # --- Tab 1: Upload ---
    with tab1:
        test_image = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"])

        if test_image is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                tmp_file.write(test_image.getbuffer())
                temp_file_path = tmp_file.name

            col1, col2 = st.columns([1, 1])

            with col1:
                st.markdown("**Input Image**")
                st.image(test_image, width="stretch")

            with st.spinner("Analyzing image..."):
                image = tf.keras.preprocessing.image.load_img(temp_file_path, target_size=IMAGE_SIZE)
                result_index, confidence = model_prediction(model, image)
                plant, condition = format_class_name(CLASS_NAMES[result_index])

            with col2:
                st.markdown("**Prediction Result**")
                st.markdown(f"""
                    <div class="result-card">
                        <div class="result-label">Plant</div>
                        <div class="result-value">{plant}</div>
                        <div class="result-label" style="margin-top:14px;">Condition</div>
                        <div class="result-value">{condition}</div>
                        <div class="result-label" style="margin-top:14px;">Confidence</div>
                        <div class="confidence-value">{confidence * 100:.2f}%</div>
                    </div>
                """, unsafe_allow_html=True)

            st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
            st.markdown("### Model Attention Map")
            st.caption("Highlighted regions indicate areas the model focused on to reach its prediction.")

            try:
                img_array = np.expand_dims(
                    tf.keras.preprocessing.image.img_to_array(image), axis=0
                )
                heatmap = make_cam_heatmap(model, img_array, LAST_CONV_LAYER_NAME)
                superimposed_img = overlay_heatmap(temp_file_path, heatmap)
                st.image(superimposed_img, width="stretch", channels="BGR")
            except ValueError:
                st.error(
                    f"Could not generate the attention map. The layer '{LAST_CONV_LAYER_NAME}' "
                    "may not match the current model architecture."
                )
            except Exception as e:
                st.error(f"An unexpected error occurred while generating the attention map: {e}")

    # --- Tab 2: Live Camera ---
    with tab2:
        st.markdown("Point your camera at a plant leaf for real-time prediction and attention visualization.")

        class VideoTransformer(VideoTransformerBase):
            def __init__(self):
                self.model = load_model()
                self.last_conv_layer_name = LAST_CONV_LAYER_NAME

            def transform(self, frame):
                img = frame.to_ndarray(format="bgr24")

                img_resized = cv2.resize(img, IMAGE_SIZE)
                img_array = np.expand_dims(img_resized, axis=0)

                predictions = self.model.predict(img_array, verbose=0)
                result_index = np.argmax(predictions)
                predicted_class = CLASS_NAMES[result_index]
                confidence = np.max(predictions)

                try:
                    heatmap = make_cam_heatmap(self.model, img_array, self.last_conv_layer_name)
                    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
                    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
                    output_img = cv2.addWeighted(img, 0.6, heatmap_colored, 0.4, 0)
                except Exception:
                    output_img = img

                label = f"{predicted_class} ({confidence * 100:.1f}%)"
                cv2.putText(
                    output_img, label, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
                )

                return output_img

        webrtc_streamer(key="live_cam_recognition", video_transformer_factory=VideoTransformer)