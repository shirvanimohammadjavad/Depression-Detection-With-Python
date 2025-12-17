Markdown

# Multi-modal Depression Detection System using AI 

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green)
![Status](https://img.shields.io/badge/Status-Completed-success)

##  Project Overview

This project is a **Multi-modal Emotion Recognition System** developed as a Bachelor's Thesis to assist in the diagnosis of depression through **Digital Phenotyping**. 

The system analyzes two primary biomarkers of mental health:
1.  **Visual Cues:** Facial expressions and micro-expressions (e.g., Flat Affect).
2.  **Vocal Cues:** Acoustic features of speech (e.g., Monotone pitch, Jitter).

By leveraging **Deep Learning (CNNs, LSTMs)** and **Signal Processing**, this tool provides an objective, non-invasive method to audit and quantify emotional states in real-time.

---

##  Key Features

* **Real-time Inference:** Live emotion detection from Webcam and Microphone using a **Flask** web application.
* **Advanced Visual Architecture:** Implementation of **Mini-Xception** (Depthwise Separable Convolutions) for lightweight, high-speed inference on edge devices.
* **Robust Audio Analysis:** Extraction of **180 acoustic features** (MFCCs, Mel-Spectrograms, Chroma) using `Librosa`.
* **Model Auditing:** Comprehensive evaluation using Confusion Matrices, Precision/Recall tracking, and Bias analysis.
* **Hyperparameter Optimization:** Systematic tuning of ML models (SVM, Random Forest) using **Grid Search** to maximize generalization.
* **Data Pipeline (ETL):** Automated scripts for data ingestion, cleaning, normalization, and augmentation.

---

## 🛠️ Tech Stack

* **Languages:** Python 3.x
* **Deep Learning:** PyTorch (Vision), TensorFlow/Keras (Audio)
* **Computer Vision:** OpenCV (`cv2`)
* **Data Analysis:** Pandas, NumPy, Scikit-learn
* **Audio Processing:** Librosa, PyAudio
* **Deployment:** Flask (Web API)
* **Visualization:** Matplotlib, Seaborn, TensorBoard
---
## Methodology
1. Visual Modality (Facial Expression Recognition)
Dataset: FER2013 (In-the-wild, 48x48 grayscale images).

Preprocessing: Histogram Equalization, Pixel Normalization [-1, 1], and Data Augmentation (Random Flip/Rotation).

Model: Mini-Xception. chosen for its efficiency. It replaces standard convolutions with depthwise separable convolutions, reducing parameters significantly (~60k params) while maintaining accuracy.

Performance: Achieved ~71% Accuracy (surpassing human-level accuracy on this dataset).

2. Audio Modality (Speech Emotion Recognition)
Datasets: RAVDESS, TESS, EMO-DB (Combined for linguistic invariance).

Feature Engineering: * MFCCs (40): Timbre and spectral envelope.

Chroma (12): Tonal content.

Mel Spectrogram (128): Energy distribution across frequencies.

Model: Comparison between LSTMs (for temporal dynamics) and optimized SVM/Random Forest classifiers.

 Evaluation & Auditing
The models were rigorously audited to ensure reliability in a clinical context:

Confusion Matrix Analysis: To identify class confusion (e.g., Sadness vs. Neutral).

Metric Tracking: Precision, Recall, and F1-Score were prioritized over simple accuracy to handle class imbalance.

Loss Curves: Monitoring Train vs. Validation loss via TensorBoard to detect and prevent Overfitting.

## Installation & Usage
Clone the repository:

Bash

git clone [https://github.com/YourUsername/Depression-Detection-AI.git](https://github.com/YourUsername/Depression-Detection-AI.git)
cd Depression-Detection-AI
Install dependencies:

Bash

pip install -r requirements.txt
Run the Real-time Vision Demo:

Bash

python camera_demo.py
Run the Web Application:

Bash

python app.py
Open your browser and navigate to http://127.0.0.1:5000/

Train the Model (Optional):

Bash

python train.py --epochs 300 --batch_size 64
## Future Work
Late Fusion: Implementing a weighted voting mechanism to mathematically combine Audio and Visual probability vectors.

Explainability (XAI): Integrating Grad-CAM to visualize ROI (Region of Interest) on faces to increase clinical trust.

DICOM Integration: Adapting the pipeline to handle medical imaging formats directly.

## Author
Mohammadjavad Shirvani

M.Sc. Student in Artificial Intelligence, FAU Erlangen-Nürnberg

Connect with me on LinkedIn


---

##  Project Structure

```text
├── checkpoint/             # Saved model weights (PyTorch/Keras)
├── data/                   # Raw and processed datasets (FER2013, RAVDESS, etc.)
├── face_detector/          # Haar Cascades & DNN face detection modules
├── model/                  # Neural Network Architectures (Mini-Xception)
├── app.py                  # Flask Web Application entry point
├── camera_demo.py          # Real-time computer vision demo script
├── create_csv.py           # ETL script: Data ingestion and CSV creation
├── data_extractor.py       # Audio feature extraction pipeline
├── dataset.py              # PyTorch Dataset class & Data Loader
├── deep_emotion_recognition.py # LSTM/GRU model definitions for Audio
├── grid_search.py          # Hyperparameter optimization script
├── train.py                # Main training loop (Epochs, Loss, Validation)
├── test.py                 # Evaluation script on test data
├── utils.py                # Helper functions (Normalization, Augmentation, Metrics)
└── requirements.txt        # Python dependencies
