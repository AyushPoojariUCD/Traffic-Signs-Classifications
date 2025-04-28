import streamlit as st
from pipeline.testing import predict_image

st.set_page_config(page_title="Traffic Sign Classification", layout="wide")

st.title("🚦 Traffic Sign Recognition App")

st.markdown("---")

# Two columns layout
col1, col2 = st.columns([1, 2])

with col1:
    uploaded_file = st.file_uploader("Upload a Traffic Sign Image", type=["png", "jpg", "jpeg"])

    predict_button = st.button("Predict")

with col2:
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

        if predict_button:
            with st.spinner('Predicting...'):
                prediction = predict_image(uploaded_file)
            st.success(f"🛑 Predicted Traffic Sign: **{prediction}**")

st.markdown("---")
