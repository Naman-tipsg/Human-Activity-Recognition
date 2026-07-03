import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import keras

# Page Config (Always First)
st.set_page_config(page_title="Human Activity Recognition", layout="wide")

# Load Model
model = tf.keras.models.load_model("model.h5")

# Load Test Data
X_test = np.load("X_test.npy")

# Load Class Map
class_map = np.load("class_map.npy", allow_pickle=True).item()

st.title("🏃 Human Activity Recognition")
st.write("Select a test sample and predict the activity.")

# Select Sample
sample_index = st.selectbox(
    "Select Test Sample",
    options=list(range(len(X_test)))
)

# Selected Sample
selected_sample = X_test[sample_index]

st.subheader("Selected Test Sample")

# Convert sample into dataframe
sample_df = pd.DataFrame(selected_sample)

st.dataframe(sample_df)

if st.button("Predict Activity"):

    sample = np.expand_dims(selected_sample, axis=0)

    prediction = model.predict(sample, verbose=0)

    pred_index = np.argmax(prediction)

    confidence = np.max(prediction) * 100

    st.success(f"Predicted Activity : {class_map[pred_index]}")

    st.write(f"Confidence : {confidence:.2f}%")