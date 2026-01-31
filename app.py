import streamlit as st
import cv2
import easyocr
import numpy as np

st.set_page_config(page_title="Camshaft OCR", layout="wide")

st.title("🔍 Camshaft Character Detection System")

# Upload image
uploaded_file = st.file_uploader(
    "Upload Camshaft Image",
    type=["jpg", "jpeg", "png"]
)

# Character format
format_choice = st.selectbox(
    "Select Character Format",
    ["Alphanumeric", "Only Numbers", "Only Letters"]
)

if uploaded_file is not None:
    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Show captured image
    st.subheader("📷 Captured Image")
    st.image(image, channels="BGR")

    # Process image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    processed = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )[1]

    st.subheader("🛠️ Processed Image")
    st.image(processed, channels="GRAY")

    # OCR
    reader = easyocr.Reader(['en'], gpu=False)
    results = reader.readtext(processed)

    extracted_text = ""
    for r in results:
        extracted_text += r[1] + " "

    # Format filtering
    if format_choice == "Only Numbers":
        extracted_text = "".join(filter(str.isdigit, extracted_text))
    elif format_choice == "Only Letters":
        extracted_text = "".join(filter(str.isalpha, extracted_text))

    st.subheader("🔤 Extracted Characters")
    st.text(extracted_text.strip())
