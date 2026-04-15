import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO

st.set_page_config(page_title="Live Emotion Detection", layout="wide")
st.title("Live Emotion Detection Pipeline")

# --- UPDATE PATH IF NEEDED ---
MODEL_PATH = r'C:\Users\clogg\Desktop\School Github Repos\Ai-Assignment\runs\classify\train3\weights\best.pt'

@st.cache_resource
def load_emotion_model():
    return YOLO(MODEL_PATH)

@st.cache_resource
def load_face_detector():
    return cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

model = load_emotion_model()
face_cascade = load_face_detector()

if model and not face_cascade.empty():
    st.sidebar.header("Controls")
    run_camera = st.sidebar.checkbox("Start Webcam")
    
    st.sidebar.markdown("---")
    st.sidebar.write("### Debugging Tools")
    # Default set to 5 based on your testing!
    pad = st.sidebar.slider("Adjust Face Padding", min_value=0, max_value=100, value=5, step=5) 
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.write("### Main Camera Feed")
        frame_window = st.image([])
        
    with col2:
        st.write("### What the AI Sees")
        crop_window = st.image([])

    if run_camera:
        cap = cv2.VideoCapture(0)
        
        while run_camera:
            ret, frame = cap.read()
            if not ret: break
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
            
            if len(faces) == 0:
                crop_window.empty()
            
            for (x, y, w, h) in faces:
                y1, y2 = max(0, y - pad), min(frame_rgb.shape[0], y + h + pad)
                x1, x2 = max(0, x - pad), min(frame_rgb.shape[1], x + w + pad)
                
                face_crop = frame_rgb[y1:y2, x1:x2]
                
                if face_crop.size != 0:
                    # Update Debug Window
                    debug_img = cv2.resize(face_crop, (100, 100))
                    crop_window.image(debug_img, width=200)

                    # --- INFERENCE & TOP 3 LOGIC ---
                    results = model(face_crop, verbose=False)
                    
                    probs = results[0].probs.data.tolist()
                    names = results[0].names
                    
                    # Sort emotions by confidence score
                    prob_dict = {names[i].upper(): probs[i] for i in range(len(names))}
                    sorted_probs = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)
                    
                    # Draw Face Bounding Box
                    cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), (0, 255, 150), 3)
                    
                    # Draw solid background for the Top 3 text (tall enough for 3 lines)
                    cv2.rectangle(frame_rgb, (x1, y1-90), (x1 + 220, y1), (0, 255, 150), -1)
                    
                    # Print the Top 3 Guesses
                    for i in range(3):
                        emo_name, emo_score = sorted_probs[i]
                        text = f"{emo_name}: {emo_score:.2f}"
                        cv2.putText(frame_rgb, text, (x1+5, y1 - 70 + (i*25)), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            
            frame_window.image(frame_rgb)
            
        cap.release()