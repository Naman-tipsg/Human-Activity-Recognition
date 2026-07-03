# 🏃 Human Activity Recognition with LSTM

A Deep Learning project that recognizes human physical activities using **Long Short-Term Memory (LSTM)** networks. The model learns temporal patterns from sensor data and predicts the activity being performed.

---

## 📌 Project Overview

Human Activity Recognition (HAR) is the process of identifying a person's activity based on data collected from wearable sensors such as accelerometers and gyroscopes.

In this project, an **LSTM (Long Short-Term Memory)** neural network is trained on sequential sensor data to classify different human activities with high accuracy.

---
<p align="center">
  <img src="model mp4.gif" alt="HAR Demo" width="700">
</p>
---
## 🚀 Features

- 📊 Human Activity Classification
- 🧠 Deep Learning using LSTM
- 📈 Data Preprocessing
- 🔄 Sequence-based Learning
- 💾 Saved Trained Model
- 🌐 Streamlit Web Application
- ⚡ Fast Activity Prediction
- 📉 Model Performance Evaluation

---

## 🛠️ Tech Stack

- Python
- TensorFlow / Keras
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Streamlit

---

## 📂 Project Structure

```
Human-Activity-Recognition/
│
├── app.py                 # Streamlit application
├── model.h5               # Trained LSTM model
├── X_test.npy             # Test dataset
├── class_map.npy          # Activity labels
├── requirements.txt
├── README.md
│
├── dataset/
│   ├── train.csv
│   ├── test.csv
│
├── notebook/
   └── Human_Activity_Recognition.ipynb

```

---

## 📊 Dataset

The project uses a Human Activity Recognition dataset containing sensor readings collected from wearable devices.

Typical activities include:

- Walking
- Walking Upstairs
- Walking Downstairs
- Sitting
- Standing
- Laying

---

## 🧠 Model Architecture

```
Input Sequence
      │
      ▼
LSTM Layer
      │
Dropout
      │
LSTM Layer
      │
Dense Layer (ReLU)
      │
Dense Layer (Softmax)
      │
Predicted Activity
```

---

## ⚙️ Installation

### Clone Repository

```bash
git clone https://github.com/yourusername/Human-Activity-Recognition.git
```

```bash
cd Human-Activity-Recognition
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run Streamlit App

```bash
streamlit run app.py
```

---

## 📈 Model Training

The model is trained using:

- Sequential Sensor Data
- LSTM Neural Network
- Adam Optimizer
- Sparse Categorical Crossentropy Loss
- Accuracy Metric

Example:

```python
model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=64,
    validation_split=0.2
)
```

---

## 📊 Evaluation Metrics

- Accuracy
- Loss
- Confusion Matrix
- Classification Report

---

## 📸 Application Preview

> Add screenshots inside the **images/** folder.

Example:

```markdown
![App Screenshot](images/output.png)
```

---

## 📦 Requirements

```
tensorflow
numpy
pandas
scikit-learn
matplotlib
streamlit
```

or

```bash
pip install tensorflow numpy pandas scikit-learn matplotlib streamlit
```

---

## 🔮 Future Improvements

- Real-time Sensor Prediction
- Mobile App Integration
- Live Activity Detection
- Model Optimization
- Deploy on Cloud
- REST API using FastAPI

---

## 👨‍💻 Author

**Naman Sharma**

- 🎓 BCA Student
- 💻 Data Science & Machine Learning Enthusiast
- 🧠 Python Developer

---

## ⭐ If you like this project

Give this repository a ⭐ on GitHub and feel free to contribute!

---

## 📜 License

This project is licensed under the MIT License.
