import streamlit as st
from PIL import Image
from ai_service import analyze_billboard, BILLBOARD_RULES
import requests
from io import BytesIO

st.set_page_config(page_title="Billboard Compliance Checker", layout="wide")

st.title("🛂 Billboard Compliance Checker")
st.write("Upload a billboard photo or provide an image URL to check compliance with rules.")

# --- Sidebar for inputs ---
st.sidebar.header("Configuration")

# State & area type selection
states = list(BILLBOARD_RULES.keys())
state = st.sidebar.selectbox("Select State", states)
area_type = st.sidebar.selectbox("Select Area Type", list(BILLBOARD_RULES[state].keys()))
visualize = st.sidebar.checkbox("Generate Annotated Detection", value=False)

# Image input
upload = st.file_uploader("Upload Billboard Image", type=["jpg", "jpeg", "png"])
url = st.text_input("...or paste an Image URL")

# --- Run analysis ---
if st.button("Analyze"):
    img_input = None
    if upload:
        img_input = Image.open(upload).convert("RGB")
    elif url:
        try:
            response = requests.get(url)
            img_input = Image.open(BytesIO(response.content)).convert("RGB")
        except Exception as e:
            st.error(f"Could not load image from URL: {e}")
    
    if img_input is None:
        st.warning("⚠️ Please provide an image.")
    else:
        with st.spinner("Analyzing billboard..."):
            result = analyze_billboard(img_input, state=state, area_type=area_type, visualize=visualize)
            analysis = result.get("analysis", {})

            # --- Display results ---
            st.subheader("📊 Analysis Result")
            status = analysis.get("legal_status", "unknown").upper()
            reason = analysis.get("reason", "")
            st.markdown(f"### ✅ Status: **{status}**")
            st.write(reason)

            col1, col2 = st.columns(2)
            with col1:
                if result.get("croppedImageUrl"):
                    st.image(result["croppedImageUrl"], caption="Cropped Billboard", use_column_width=True)
            with col2:
                if result.get("visualized_image"):
                    st.image(result["visualized_image"], caption="Annotated Detection", use_column_width=True)

            st.subheader("🔍 Raw Data")
            st.json(result)
