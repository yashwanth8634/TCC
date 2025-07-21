import streamlit as st
import numpy as np
import tensorflow as tf

# Set page config
st.set_page_config(page_title="TCC Patch Classifier 📊", layout="centered", page_icon="🌩")

# Custom CSS for professional look
st.markdown("""
    <style>
    .main {background-color: #F9FAFB;}
    .stButton>button {background-color:#384B9B; color:white;}
    .reportview-container .main footer {visibility: hidden;}
    .stProgress > div > div > div > div {background-color: #384B9B;}
    </style>
""", unsafe_allow_html=True)

# Sidebar for about/project info
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/5/5b/Logo_of_CVR_College_of_Engineering.png", width=130)
    st.title("About Project")
    st.markdown("""
    **Tropical Cloud Cluster Patch Classifier**  
    Developed at CVR College of Engineering  
    Use this tool to classify 32x32 grayscale satellite patches as TCC/non-TCC.
    """)
    st.markdown("---")
    st.info("📢 Ensure your input is a *.npy* file of shape *(32,32)* or *(32,32,1)*.")

# App Title and Instructions
st.title("🌩 TCC Patch Classifier")
st.markdown(
    "<span style='font-size:17px;'>Quickly identify if a 32x32 grayscale patch represents a <b>Tropical Cloud Cluster</b> from satellite data.</span>",
    unsafe_allow_html=True,
)
st.markdown("")

instructions = """
#### 📝 How to use:
- Step 1: Download a [sample file](https://drive.google.com/uc?export=download&id=<your_sample_file_id>) (if needed)
- Step 2: Click 'Upload Patch File' and select your *32x32 .npy* file
- Step 3: See the prediction and confidence below
"""
with st.expander("Show Instructions"):
    st.markdown(instructions, unsafe_allow_html=True)
# Add this function to your app.py
def dice_coefficient(y_true, y_pred, smooth=1):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    return (2. * intersection + smooth) / (K.sum(K.square(y_true),-1) + K.sum(K.square(y_pred),-1) + smooth)
# Load trained model
@st.cache_resource(show_spinner=True)
def load_model():
    return tf.keras.models.load_model("tcc_unet_classifier.h5",custom_objects={"dice_coefficient": dice_coefficient})

model = load_model()

# File uploader
st.markdown("#### 📁 Upload a 32x32 grayscale `.npy` patch:")
uploaded_file = st.file_uploader("", type=["npy"])

if uploaded_file:
    with st.spinner("Analyzing patch..."):
        try:
            patch = np.load(uploaded_file)
            if patch.ndim == 2:
                patch = patch[..., np.newaxis]
            if patch.shape != (32, 32, 1):
                st.error(f"❌ Invalid patch shape: {patch.shape}. Expected (32, 32, 1)")
            else:
                # Normalize patch for display
                patch_display = patch[:, :, 0].astype(np.float32)
                if patch_display.max() > 1.0 or patch_display.min() < 0.0:
                    patch_display = (patch_display - patch_display.min()) / (patch_display.max() - patch_display.min())
                patch_display = np.clip(patch_display, 0.0, 1.0)

                # Predict
                pred = model.predict(np.expand_dims(patch, axis=0), verbose=0)[0][0]
                label = "TCC 🌩" if pred > 0.5 else "Non-TCC ☀"
                confidence = pred if pred > 0.5 else 1 - pred

                # Color for result display
                color = "#47c479" if pred > 0.5 else "#e98b1c"

                st.image(patch_display, width=160, caption="🖼 Uploaded Patch", channels="GRAY")
                st.markdown(f"""
                <div style="background-color:{color}; color:white; padding:10px; border-radius:5px; text-align:center; font-size:18px;">
                🧠 Prediction: <b>{label}</b>
                <hr style="margin:10px 0; border:0.5px solid #dee2e6;">
                🔢 <b>Confidence:</b> {confidence:.2%}  
                🎯 <b>Raw Score:</b> {pred:.3f}
                </div>
                """, unsafe_allow_html=True)
                st.success("🎉 Prediction completed!")

        except Exception as e:
            st.error(f"⚠ Error processing file: {e}")

else:
    st.info("⬆️ Upload a .npy patch file to begin.")    

# Footer
st.markdown("---")
st.markdown(
    "<center><small>Made with ❤️ by the Department of IT, CVR COLLEGE OF ENGINEERING</small></center>",
    unsafe_allow_html=True,
)
