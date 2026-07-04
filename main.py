import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import tempfile
import os
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

from remedies import get_remedy

try:
    from google import genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

# ----------------------------------------------------------------------------
# Page Configuration
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Plant Disease Recognition System",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------------------------------------------------------
# Custom Styling — Green / Plant-Care Theme
#
# NOTE: this CSS block styles content Streamlit lets us reach with markup.
# It intentionally does NOT try to re-theme the header bar, toolbar, file
# uploader chrome, or widget internals — those are rendered by Streamlit
# itself using .streamlit/config.toml, which is the only reliable way to
# theme them (inline CSS selectors for that chrome break on every Streamlit
# version bump). Ship both files together.
# ----------------------------------------------------------------------------
st.markdown("""
    <style>
        /* ------------------------------------------------------------------
           Palette — anchored on Spanish Green (#009150)
           - Accent:      #009150  (buttons, active states, badges)
           - Accent dark: #00693c  (headings, hover)
           - Accent deep: #004d2c  (sidebar active dot, strong emphasis)
           - Tint 1:      #e6f7ec  (sidebar bg / subtle panels)
           - Tint 2:      #d1efdc  (borders, dividers)
           - Body text:   #1c2b22  (near-black green-grey, safe contrast)
        ------------------------------------------------------------------ */
        .stApp {
            background-color: #eaf7ee;
        }
        section[data-testid="stSidebar"] {
            background-color: #d4efdc;
            border-right: 1px solid #a9dfba;
        }

        /* Headings everywhere, INCLUDING inside the sidebar. The sidebar's
           own theme classes can out-specificity a bare "h1" rule, which is
           why "Navigation" was rendering invisible — these selectors target
           the sidebar explicitly and win. */
        h1, h2, h3,
        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3 {
            font-weight: 700;
            letter-spacing: -0.5px;
            color: #00522f !important;
        }
        p, li, span, label, strong, b, .stCaption, [data-testid="stCaptionContainer"],
        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] span,
        section[data-testid="stSidebar"] strong,
        section[data-testid="stSidebar"] label {
            color: #12261a !important;
        }
        section[data-testid="stSidebar"] .stCaption,
        section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
            color: #3d6b4e !important;
        }

        /* Sidebar nav radio — active item gets the accent dot + bold label */
        section[data-testid="stSidebar"] [role="radiogroup"] label {
            font-weight: 600;
        }

        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 20px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 44px;
            padding: 0 8px;
            font-weight: 600;
            color: #2f6b48;
        }
        .stTabs [aria-selected="true"] {
            color: #00522f !important;
            border-bottom-color: #009150 !important;
        }

        /* File uploader card */
        [data-testid="stFileUploaderDropzone"] {
            background-color: #f4fbf6;
            border: 1.5px dashed #6fc491;
            border-radius: 14px;
        }

        /* Selectbox / radio */
        [data-baseweb="select"] > div {
            border-radius: 10px;
            border-color: #a9dfba !important;
        }
        .stRadio [role="radiogroup"] label {
            color: #12261a;
        }

        /* Alerts (info/warning/error) rounded to match the card language */
        div[data-testid="stAlert"] {
            border-radius: 12px;
        }

        /* Card components */
        .result-card {
            background-color: #f6fdf8;
            border: 1px solid #a9dfba;
            border-radius: 16px;
            padding: 22px 26px;
            margin-top: 12px;
            box-shadow: 0 2px 10px rgba(0, 82, 47, 0.10);
        }
        .result-label {
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: #2f7d54;
            margin-bottom: 4px;
            font-weight: 600;
        }
        .result-value {
            font-size: 22px;
            font-weight: 700;
            color: #00522f;
        }
        .confidence-value {
            font-size: 16px;
            font-weight: 700;
            color: #009150;
        }
        .rec-card {
            background: linear-gradient(180deg, #f6fdf8 0%, #dff2e6 100%);
            border: 1px solid #a9dfba;
            border-left: 5px solid #009150;
            border-radius: 14px;
            padding: 22px 26px;
            margin-top: 16px;
            line-height: 1.6;
            color: #12261a;
        }
        .rec-title {
            font-size: 14px;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: #009150;
            font-weight: 700;
            margin-bottom: 10px;
        }
        .rec-badge {
            display: inline-block;
            font-size: 11px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            padding: 3px 10px;
            border-radius: 999px;
            margin-left: 10px;
            vertical-align: middle;
        }
        .rec-badge-gemini {
            background-color: #bce8cd;
            color: #00522f;
        }
        .rec-badge-fallback {
            background-color: #fdf1d6;
            color: #8a6d1a;
        }
        .severity-High { color: #b3261e; font-weight: 700; }
        .severity-Medium { color: #b8860b; font-weight: 700; }
        .severity-Low { color: #009150; font-weight: 700; }
        .severity-None { color: #009150; font-weight: 700; }
        .feature-card {
            background-color: #f6fdf8;
            border: 1px solid #a9dfba;
            border-radius: 14px;
            padding: 18px 20px;
            height: 100%;
        }
        .section-divider {
            border-top: 1px solid #a9dfba;
            margin: 24px 0;
        }

        /* Buttons */
        .stButton > button {
            background-color: #009150;
            color: white;
            border-radius: 10px;
            border: none;
            font-weight: 600;
            padding: 8px 20px;
        }
        .stButton > button:hover {
            background-color: #00522f;
            color: white;
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
GEMINI_MODEL_NAME = "gemini-2.5-flash"

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

LANGUAGE_OPTIONS = {
    "English": "English",
    "हिंदी (Hindi)": "Hindi",
    "తెలుగు (Telugu)": "Telugu",
    "தமிழ் (Tamil)": "Tamil",
    "ಕನ್ನಡ (Kannada)": "Kannada",
    "മലയാളം (Malayalam)": "Malayalam",
    "मराठी (Marathi)": "Marathi",
    "ગુજરાતી (Gujarati)": "Gujarati",
    "বাংলা (Bengali)": "Bengali",
    "ਪੰਜਾਬੀ (Punjabi)": "Punjabi",
    "ଓଡ଼ିଆ (Odia)": "Odia",
    "অসমীয়া (Assamese)": "Assamese",
    "اردو (Urdu)": "Urdu",
    "कोंकणी (Konkani)": "Konkani",
    "मैथिली (Maithili)": "Maithili",
    "नेपाली (Nepali)": "Nepali",
    "संस्कृतम् (Sanskrit)": "Sanskrit",
    "بودو (Bodo)": "Bodo",
    "کٲشُر (Kashmiri)": "Kashmiri",
    "डोगरी (Dogri)": "Dogri",
    "संताली (Santali)": "Santali",
    "মৈতৈলোন্ (Manipuri)": "Manipuri",
    "سنڌي (Sindhi)": "Sindhi",
}


# ----------------------------------------------------------------------------
# Model Loading
# ----------------------------------------------------------------------------
@st.cache_resource()
def load_model():
    """Loads and caches the trained Keras model."""
    return tf.keras.models.load_model(MODEL_PATH)


# ----------------------------------------------------------------------------
# Gemini configuration + diagnostics
#
# Design goal: never let a silent/ambiguous "not configured" state ship.
# get_gemini_status() always returns a precise reason, and the UI (sidebar
# banner + on-page banner) shows it, with concrete next steps.
# ----------------------------------------------------------------------------
def get_gemini_api_key():
    """Fetches the Gemini API key from Streamlit secrets or environment variables."""
    key = None
    try:
        # st.secrets raises if no secrets.toml exists at all in some
        # Streamlit versions — guard the whole lookup, not just the "in" check.
        if "GEMINI_API_KEY" in st.secrets:
            key = st.secrets["GEMINI_API_KEY"]
    except Exception:
        key = None

    if not key:
        key = os.getenv("GEMINI_API_KEY")

    if key is not None:
        key = key.strip()

    return key or None


def get_gemini_status():
    """
    Returns a dict describing exactly why Gemini is or isn't available:
    {"ok": bool, "reason": str, "message": str}
    reason is one of: "ok", "package_missing", "missing_key", "client_error"
    """
    if not GENAI_AVAILABLE:
        return {
            "ok": False,
            "reason": "package_missing",
            "message": (
                "The `google-genai` package isn't installed. Add `google-genai` "
                "to requirements.txt and redeploy."
            ),
        }

    api_key = get_gemini_api_key()
    if not api_key:
        return {
            "ok": False,
            "reason": "missing_key",
            "message": (
                "No `GEMINI_API_KEY` was found in Streamlit secrets or environment "
                "variables. On Streamlit Cloud: App settings → Secrets, and add "
                "`GEMINI_API_KEY = \"your-key-here\"`. Locally: create "
                "`.streamlit/secrets.toml` with the same line, or set the "
                "`GEMINI_API_KEY` environment variable before running the app."
            ),
        }

    client, err = _build_gemini_client(api_key)
    if client is None:
        return {
            "ok": False,
            "reason": "client_error",
            "message": f"A `GEMINI_API_KEY` was found, but the client failed to initialize: {err}",
        }

    return {"ok": True, "reason": "ok", "message": "Gemini is configured and ready."}


@st.cache_resource(show_spinner=False)
def _build_gemini_client(api_key):
    """
    Builds and caches a Gemini client, keyed by the api_key value itself.
    Keying the cache on the key (rather than on nothing) means a key added or
    changed in secrets during local development is picked up on the next
    rerun, instead of being stuck behind a stale cached None.
    """
    try:
        return genai.Client(api_key=api_key), None
    except Exception as e:
        return None, str(e)


def get_gemini_client():
    """Returns a ready-to-use Gemini client, or None if not available."""
    api_key = get_gemini_api_key()
    if not api_key or not GENAI_AVAILABLE:
        return None
    client, _ = _build_gemini_client(api_key)
    return client


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


def render_fallback_html(remedy, language_note=None):
    """Renders the built-in remedy dict as HTML for the rec-card, in English."""
    parts = [f"<p>{remedy['summary']}</p>"]

    if not remedy["is_healthy"]:
        parts.append(
            f"<p><strong>Severity:</strong> "
            f"<span class='severity-{remedy['severity']}'>{remedy['severity']}</span></p>"
        )
        parts.append("<p><strong>Organic / cultural treatment:</strong></p><ul>")
        parts += [f"<li>{item}</li>" for item in remedy["organic"]]
        parts.append("</ul>")
        parts.append("<p><strong>Chemical treatment options:</strong></p><ul>")
        parts += [f"<li>{item}</li>" for item in remedy["chemical"]]
        parts.append("</ul>")

    parts.append("<p><strong>Prevention tips:</strong></p><ul>")
    parts += [f"<li>{item}</li>" for item in remedy["prevention"]]
    parts.append("</ul>")

    if language_note:
        parts.append(f"<p style='font-size:13px;color:#6b8f7a;margin-top:10px;'>{language_note}</p>")

    return "".join(parts)


def build_recommendation_prompt(plant, condition, confidence, language, remedy):
    """
    Builds the prompt sent to Gemini. Rather than asking Gemini to invent
    treatment advice from scratch, the built-in remedy reference is supplied
    as grounding context, and Gemini's job is to personalize the tone/length
    for a farmer and translate it into the requested language. This keeps
    the underlying agronomic facts stable and app-controlled, while still
    getting Gemini's language and phrasing quality.
    """
    is_healthy = remedy["is_healthy"]

    reference_lines = [f"Summary: {remedy['summary']}"]
    if not is_healthy:
        reference_lines.append(f"Severity: {remedy['severity']}")
        reference_lines.append("Organic treatment: " + "; ".join(remedy["organic"]))
        reference_lines.append("Chemical treatment: " + "; ".join(remedy["chemical"]))
    reference_lines.append("Prevention: " + "; ".join(remedy["prevention"]))
    reference_text = "\n".join(reference_lines)

    if is_healthy:
        task = (
            f"The plant is a {plant} and it has been classified as HEALTHY "
            f"with {confidence * 100:.1f}% confidence. Using the reference notes below, "
            "write a short, reassuring note confirming the plant looks healthy, plus "
            "2-3 general preventive care tips to keep it that way."
        )
    else:
        task = (
            f"A farmer has uploaded a photo of a {plant} leaf. A deep learning model has "
            f"diagnosed it with '{condition}' at {confidence * 100:.1f}% confidence. Using the "
            "reference notes below as your factual basis (do not contradict them or invent "
            "different treatments), write a short, practical, farmer-facing recommendation with "
            "these sections: 1) What this disease is (1-2 simple sentences, no jargon), "
            "2) Severity/urgency level, 3) Immediate treatment steps (organic and chemical options), "
            "4) Prevention tips for future crops."
        )

    return (
        f"{task}\n\nReference notes (factual basis, in English):\n{reference_text}\n\n"
        f"Respond entirely in {language}. Keep the tone simple, clear, and practical "
        "for a farmer with no technical background. Use short sentences and avoid "
        "markdown headers with # symbols; use plain text with simple line breaks and "
        "numbered/bulleted lists where helpful. Keep the total response under 200 words."
    )


def get_recommendation(plant, condition, confidence, language, class_name):
    """
    Always returns usable content for the UI:
    (html_body, source, banner_message)
      source is "gemini" or "fallback"
      banner_message is None, or a short st.info/st.warning-worthy string
    """
    remedy = get_remedy(class_name)
    status = get_gemini_status()

    if not status["ok"]:
        html_body = render_fallback_html(
            remedy,
            language_note=(
                None if language == "English" else
                "Gemini is not configured, so this reference is shown in English only. "
                "Configure a GEMINI_API_KEY to get this translated and personalized."
            ),
        )
        return html_body, "fallback", status["message"]

    client = get_gemini_client()
    prompt = build_recommendation_prompt(plant, condition, confidence, language, remedy)

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL_NAME,
            contents=prompt,
        )
        text = (response.text or "").strip()
        if not text:
            raise ValueError("Gemini returned an empty response.")
        html_body = text.replace(chr(10), "<br>")
        return html_body, "gemini", None
    except Exception as e:
        err_str = str(e).lower()
        if "api key" in err_str or "permission" in err_str or "unauthenticated" in err_str or "401" in err_str or "403" in err_str:
            banner = f"Your GEMINI_API_KEY appears to be invalid or unauthorized ({e}). Showing the built-in reference instead."
        else:
            banner = f"Could not reach Gemini right now ({e}). Showing the built-in reference instead."
        html_body = render_fallback_html(remedy)
        return html_body, "fallback", banner


# ----------------------------------------------------------------------------
# Sidebar Navigation
# ----------------------------------------------------------------------------
st.sidebar.title(" Navigation")
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

st.sidebar.markdown("**AI Recommendations**")
_gemini_status = get_gemini_status()
if _gemini_status["ok"]:
    st.sidebar.caption(" Powered by Gemini API")
else:
    st.sidebar.caption(" Not configured — built-in reference used instead")
    with st.sidebar.expander("Why, and how to fix it"):
        st.write(_gemini_status["message"])


# ----------------------------------------------------------------------------
# Home Page
# ----------------------------------------------------------------------------
if app_mode == "Home":
    st.title(" Plant Disease Recognition System")
    st.caption("Deep learning-based leaf disease identification for crop health monitoring")

    if os.path.exists("home_page.jpeg"):
        st.image("home_page.jpeg", width="stretch")
    else:
        st.info("Home page image not found. Place 'home_page.jpeg' in the project directory to display it here.")

    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
            <div class="feature-card">
                <strong> Upload & Detect</strong>
                <p style="margin-top:8px; font-size:14px;">Upload a leaf image and get an instant diagnosis with a confidence score.</p>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
            <div class="feature-card">
                <strong> Model Explainability</strong>
                <p style="margin-top:8px; font-size:14px;">View a Class Activation Map showing which regions influenced the prediction.</p>
            </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
            <div class="feature-card">
                <strong> AI Recommendations</strong>
                <p style="margin-top:8px; font-size:14px;">Get farmer-friendly treatment advice, generated by Gemini when configured, or from a built-in reference otherwise.</p>
            </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
            <div class="feature-card">
                <strong> Live Camera Mode</strong>
                <p style="margin-top:8px; font-size:14px;">Run real-time detection directly from a webcam feed.</p>
            </div>
        """, unsafe_allow_html=True)

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
        "provides both a prediction and a visual explanation of the model's decision. Every "
        "diagnosis is paired with a treatment reference: a built-in organic/chemical/prevention "
        "guide for all 38 classes that always works, personalized and translated by Gemini "
        "into the farmer's preferred language whenever a Gemini API key is configured."
    )

    st.markdown("### Model Details")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
            <div class="result-card">
                <div class="result-label">Architecture</div>
                <div class="result-value">CNN (TensorFlow / Keras)</div>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
            <div class="result-card">
                <div class="result-label">Output Classes</div>
                <div class="result-value">{len(CLASS_NAMES)}</div>
            </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
            <div class="result-card">
                <div class="result-label">Recommendations</div>
                <div class="result-value">Built-in + Gemini</div>
            </div>
        """, unsafe_allow_html=True)

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

    if not _gemini_status["ok"]:
        st.warning(
            " Gemini isn't configured right now, so recommendations below will use the "
            "app's built-in treatment reference in English instead of a personalized, "
            "translated one. See the sidebar for how to fix this."
        )

    model = load_model()

    tab1, tab2 = st.tabs(["Upload Image", "Live Camera"])

    # --- Tab 1: Upload ---
    with tab1:
        upload_col, lang_col = st.columns([3, 1])
        with upload_col:
            test_image = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"])
        with lang_col:
            selected_language_label = st.selectbox(
                "Recommendation language",
                list(LANGUAGE_OPTIONS.keys()),
            )
            selected_language = LANGUAGE_OPTIONS[selected_language_label]

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
                class_name = CLASS_NAMES[result_index]
                plant, condition = format_class_name(class_name)

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

            with st.spinner("Preparing recommendation..."):
                rec_html, rec_source, rec_banner = get_recommendation(
                    plant, condition, confidence, selected_language, class_name
                )

            badge_class = "rec-badge-gemini" if rec_source == "gemini" else "rec-badge-fallback"
            badge_label = "Gemini" if rec_source == "gemini" else "Built-in reference"

            st.markdown(
                f"###  Treatment Recommendation "
                f"<span class='rec-badge {badge_class}'>{badge_label}</span>",
                unsafe_allow_html=True,
            )
            st.caption("Actionable guidance based on the diagnosis above.")

            if rec_banner:
                st.info(rec_banner)

            st.markdown(f"""
                <div class="rec-card">
                    <div class="rec-title">Recommendation ({selected_language})</div>
                    {rec_html}
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