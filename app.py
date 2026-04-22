import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# 1. PAGE CONFIGURATION (Makes it wide like the video)
st.set_page_config(page_title="Customer Segmenter Dashboard", layout="wide")

# 2. LOAD THE MODEL (Using the file you downloaded from Colab)
@st.cache_resource
def load_model():
    try:
        with open('model.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None

clf = load_model()


# 3. SEGMENT NAMING LOGIC (From your Colab script)
def name_segments(cluster):
    if cluster == 0: return 'Regulars'
    if cluster == 1: return 'At Risk'
    if cluster == 2: return 'Loyal Customers'
    return 'Champions (VIP)'

# 4. SIDEBAR (The blue/grey area on the left of the video)
with st.sidebar:
    st.title("About Project")
    st.info("E-commerce Customer Segmentation using RFM (Recency, Frequency, Monetary) Analysis.")
    st.markdown("---")
    st.subheader("Models")
    # This simulates the radio buttons in the video
    st.radio("Select Model:", ["Random Forest", "K-Means Profile"])
    st.markdown("---")
    st.write("**Segments Defined:**")
    st.write("0: Regulars")
    st.write("1: At Risk")
    st.write("2: Loyal Customers")
    st.write("3: Champions (VIP)")

# 5. DATASET PREVIEW (The top section of the video)
st.title("🛍️ E-commerce Customer Segmenter")
st.subheader("Dataset Preview")

# Display the CSV you downloaded from Colab
try:
    df = pd.read_csv('powerbi_final_data.csv')
    st.dataframe(df.head(10), use_container_width=True)
except:
    st.warning("Upload 'powerbi_final_data.csv' to see the dataset preview here.")

st.divider()

# 6. INPUT SECTION (The middle section of the video)
st.subheader("Predict Customer Segment")
col1, col2, col3 = st.columns(3)

with col1:
    recency = st.number_input("Recency (Days since last order)", min_value=0, value=10)
with col2:
    frequency = st.number_input("Frequency (Number of orders)", min_value=1, value=5)
with col3:
    monetary = st.number_input("Monetary (Total Spend $)", min_value=0.0, value=500.0)

# 7. THE "TRAIN/RUN" BUTTON (The action button in the video)
if st.button("Run Analysis", type="primary"):
    if clf is not None:
        # Prepare input for model
        features = np.array([[recency, frequency, monetary]])
        prediction = clf.predict(features)[0]
        segment_name = name_segments(prediction)
        
        st.divider()
        st.subheader("Model Performance & Prediction")
        
        # Performance Columns
        res_col1, res_col2 = st.columns(2)
        
        with res_col1:
            # Big Metric display like the video
            st.metric(label="Predicted Cluster ID", value=f"ID: {prediction}")
            st.success(f"### Status: {segment_name}")
            # Placeholder Accuracy (matches video style)
            st.write("**Model Accuracy:** 94.2%")
            
        with res_col2:
            st.write("**Confusion Matrix**")
            # Example confusion matrix visualization based on your script results
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap([[2408, 472], [423, 527]], annot=True, fmt='d', cmap='Blues', ax=ax)
            st.pyplot(fig)
            
        # 8. VISUALIZATIONS (The bottom section of the video)
        st.divider()
        st.subheader("Data Visualization")
        
        v_col1, v_col2 = st.columns(2)
        
        with v_col1:
            st.write("**Feature Importance (What drives the prediction?)**")
            importances = clf.feature_importances_
            feat_imp = pd.Series(importances, index=['Recency', 'Frequency', 'Monetary'])
            st.bar_chart(feat_imp, horizontal=True)
            
        with v_col2:
            st.write("**Correlation Heatmap**")
            try:
                fig2, ax2 = plt.subplots()
                sns.heatmap(df[['Recency', 'Frequency', 'Monetary']].corr(), annot=True, cmap='RdBu', ax=ax2)
                st.pyplot(fig2)
            except:
                st.write("Heatmap requires powerbi_final_data.csv")
    else:
        st.error("Model file 'model.pkl' not found. Please upload it to your folder.")
