# Ai-Assignment
Facial Emotion Recognition System - Group 4 (RSW Y2S2)
Student Name: Eizen Lim Hoe Yuen

Project: Emotion Detection AI via Traditional ML and Deep Learning architectures.

📂 Folder Structure & Submission Guide
1. Proof_we_trained_our_models
This folder contains the verifiable "paper trail" for every model developed in this study.

YOLOv8 Logs: Contains the runs/ folders for Experiments 1 through 4 (Baseline, Augmentation, Preprocessing, and Oversampled).

Traditional ML (SVM) Proof: Unlike YOLO, the SVM does not generate native "run" folders. Proof of training is provided via:

svm_accuracy_flow.png: Visualizes the learning progress across 33,404 samples.

svm_confusion_matrix.png: Demonstrates the model's classification accuracy on the test set.

Classification Report: Detailed Precision, Recall, and F1-score metrics.

2. Notebooks
Contains the source code and execution history for all models.

main.ipynb: The primary notebook for SVM feature extraction (HOG+LBP) and training. Note: All cell outputs have been left visible to verify the 86% accuracy achievement.

YOLO & CNN Notebooks: Source scripts for the Deep Learning architectures.

3. Models
This folder contains the final "brains" of our application.

best_oversampled.pt: The champion YOLOv8 model weights.

emotion_model_final.pkl (1.2GB): The fully trained SVM model. Its size is a result of storing high-dimensional support vectors derived from 4,327 PCA components.

cnn_emotion_model.h5: The trained MobileNetV2 weights.

4. Prototype_App
The production-ready Streamlit application.

app.py: The main execution script.

Functional Modes: Supports real-time webcam monitoring (via MediaPipe 3D Face Mesh) and static batch image uploads.

🛠️ How to Verify the TML (SVM) Model
As the SVM does not produce a YOLO-style runs folder, its validity can be confirmed by:

Loading the .pkl: The 1.2GB file in Folder 3 can be loaded via joblib to replicate the results in the report.

Notebook Outputs: The main.ipynb in Folder 2 contains the logged extraction process for all 33,404 training samples.