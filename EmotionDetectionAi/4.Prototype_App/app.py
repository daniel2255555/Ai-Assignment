import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import mediapipe as mp
import pandas as pd
import time
from pathlib import Path

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Emotion Recognition - YOLO Demo", 
    layout="wide", 
    page_icon="🎭"
)

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 25px;
        border-radius: 10px;
        text-align: center;
        color: white;
        margin-bottom: 20px;
    }
    .stAlert {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 15px;
        border-radius: 5px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <div class="main-header">
        <h1>🎭 Facial Emotion Recognition</h1>
        <p>BMCS2203 AI Assignment | Group 4 | YOLOv8m-cls (Oversampled) - 88% Accuracy</p>
    </div>
""", unsafe_allow_html=True)

# --- LOAD MODELS ---
@st.cache_resource
def load_yolo_model():
    # Go up one directory from app.py location, then into 3.Models folder
    model_path = Path(__file__).parent.parent / "3.Models" / "best_oversampled.pt"
    
    if not model_path.exists():
        st.error(f"❌ Model not found at: {model_path}")
        st.info("💡 Looking for model at: C:\\Users\\clogg\\Desktop\\School Github Repos\\Ai-Assignment\\EmotionDetectionAi\\3.Models\\best_oversampled.pt")
        st.stop()
    
    return YOLO(str(model_path))

@st.cache_resource
def load_face_detector():
    mp_face_mesh = mp.solutions.face_mesh
    return mp_face_mesh.FaceMesh(
        max_num_faces=5, 
        min_detection_confidence=0.5, 
        min_tracking_confidence=0.5
    )

model = load_yolo_model()
face_mesh = load_face_detector()

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    
    # Feature toggles
    show_top3 = st.checkbox("Show Top 3 Predictions", value=True)
    drowsiness_alert = st.checkbox("🚗 Driver Drowsiness Alert", value=False)
    
    if drowsiness_alert:
        alert_duration = st.slider("Alert after (seconds):", 3, 15, 5)
        st.info("💡 Simulates driver monitoring system - alerts if 'Sad' or 'Neutral' detected for too long")
    
    st.markdown("---")
    st.markdown("### 📊 Model Info")
    st.metric("Model", "YOLOv8m-cls")
    st.metric("Accuracy", "88%")
    st.metric("Dataset", "RAF-DB")
    st.metric("Inference Speed", "~60 FPS")

# --- TABS ---
tab1, tab2 = st.tabs(["🎥 Live Demo", "📊 Research Results"])

# ==========================================
# TAB 1: LIVE DEMO
# ==========================================
with tab1:
    st.markdown("## 🎬 Real-Time Emotion Detection")
    
    input_mode = st.radio(
        "Select Input Mode:", 
        ["📷 Image Upload", "🎥 Live Webcam"], 
        horizontal=True
    )
    
    st.markdown("---")
    
    def get_top3_predictions(results):
        """Extract top 3 predictions with probabilities"""
        probs = results[0].probs.data.cpu().numpy()
        top3_indices = np.argsort(probs)[-3:][::-1]
        
        predictions = []
        for idx in top3_indices:
            emotion = results[0].names[idx]
            confidence = probs[idx] * 100
            predictions.append((emotion, confidence))
        
        return predictions
    
    def process_face_with_mesh(frame, results_mesh, model):
        """Process faces using MediaPipe 3D Face Mesh"""
        faces_data = []
        
        if results_mesh.multi_face_landmarks:
            h, w, _ = frame.shape
            for face_landmarks in results_mesh.multi_face_landmarks:
                
                # Extract bounding box from 468 landmarks
                x_coords = [landmark.x * w for landmark in face_landmarks.landmark]
                y_coords = [landmark.y * h for landmark in face_landmarks.landmark]
                
                x_min, x_max = int(min(x_coords)), int(max(x_coords))
                y_min, y_max = int(min(y_coords)), int(max(y_coords))
                
                box_w = x_max - x_min
                box_h = y_max - y_min
                
                # Add padding
                pad_x, pad_y = int(box_w * 0.15), int(box_h * 0.15)
                x = max(0, x_min - pad_x)
                y = max(0, y_min - pad_y)
                x_end = min(w, x_max + pad_x)
                y_end = min(h, y_max + pad_y)
                
                # Crop face
                face_roi = frame[y:y_end, x:x_end]
                
                if face_roi.size == 0:
                    continue
                
                # YOLO prediction
                results = model(face_roi, verbose=False)
                top3 = get_top3_predictions(results)
                top_emotion, top_conf = top3[0]
                
                faces_data.append({
                    'bbox': (x, y, x_end, y_end),
                    'top_emotion': top_emotion,
                    'top_conf': top_conf,
                    'top3': top3
                })
                
                # Draw bounding box
                cv2.rectangle(frame, (x, y), (x_end, y_end), (102, 126, 234), 2)
                
                # Label
                label = f"{top_emotion.upper()} {top_conf:.1f}%"
                cv2.rectangle(frame, (x, y-30), (x + len(label)*12, y), (102, 126, 234), -1)
                cv2.putText(frame, label, (x+5, y-8), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame, faces_data
    
    # ==========================================
    # MODE 1: IMAGE UPLOAD
    # ==========================================
    if input_mode == "📷 Image Upload":
        uploaded_file = st.file_uploader(
            "Upload an image (JPG, PNG)", 
            type=['jpg', 'jpeg', 'png']
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            with st.spinner('🔍 Analyzing emotions...'):
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mesh_results = face_mesh.process(frame_rgb)
                
                annotated_frame, faces_data = process_face_with_mesh(frame, mesh_results, model)
                rgb_final = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                
                if not faces_data:
                    st.warning("⚠️ No faces detected")
                    st.image(image, use_container_width=True)
                else:
                    # Display image and results side by side
                    col_img, col_results = st.columns([2, 1])
                    
                    with col_img:
                        st.image(rgb_final, caption="✅ Detection Complete", use_container_width=True)
                    
                    with col_results:
                        st.markdown("### 🎯 Detection Results")
                        
                        for i, face in enumerate(faces_data, 1):
                            st.markdown(f"**Face {i}:**")
                            
                            if show_top3:
                                # Show top 3 predictions
                                for rank, (emotion, conf) in enumerate(face['top3'], 1):
                                    emoji = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉"
                                    st.write(f"{emoji} {emotion.title()}: {conf:.1f}%")
                            else:
                                # Show only top prediction
                                st.write(f"🎯 {face['top_emotion'].title()}: {face['top_conf']:.1f}%")
                            
                            st.markdown("---")
    
    # ==========================================
    # MODE 2: LIVE WEBCAM
    # ==========================================
    elif input_mode == "🎥 Live Webcam":
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            run_webcam = st.checkbox("▶️ Start Webcam", value=False)
        
        with col2:
            if run_webcam:
                st.markdown("🔴 **LIVE**")
        
        FRAME_WINDOW = st.empty()
        ALERT_PLACEHOLDER = st.empty()
        
        # Drowsiness detection state
        drowsy_start_time = None
        alert_playing = False
        
        if run_webcam:
            cap = cv2.VideoCapture(0)
            
            while run_webcam:
                ret, frame = cap.read()
                if not ret:
                    st.error("❌ Failed to access webcam")
                    break
                
                # Process frame
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mesh_results = face_mesh.process(frame_rgb)
                
                annotated_frame, faces_data = process_face_with_mesh(frame, mesh_results, model)
                
                # Drowsiness detection logic
                if drowsiness_alert and faces_data:
                    # Check if any face shows drowsy emotion
                    drowsy_emotions = ['sad', 'neutral']
                    is_drowsy = any(
                        face['top_emotion'].lower() in drowsy_emotions 
                        for face in faces_data
                    )
                    
                    if is_drowsy:
                        if drowsy_start_time is None:
                            drowsy_start_time = time.time()
                        
                        elapsed = time.time() - drowsy_start_time
                        
                        # Show countdown
                        if elapsed < alert_duration:
                            with ALERT_PLACEHOLDER:
                                remaining = alert_duration - int(elapsed)
                                st.warning(f"⚠️ Drowsiness detected: Alert in {remaining}s...")
                        else:
                            # ALERT!
                            with ALERT_PLACEHOLDER:
                                st.error("🚨 DRIVER ALERT: Take a break! Pull over safely.")
                                # Could add actual sound here with st.audio()
                                if not alert_playing:
                                    st.balloons()  # Visual alert
                                    alert_playing = True
                    else:
                        drowsy_start_time = None
                        alert_playing = False
                        ALERT_PLACEHOLDER.empty()
                
                # Display frame
                rgb_final = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                FRAME_WINDOW.image(rgb_final, channels="RGB", use_container_width=True)
            
            cap.release()

# ==========================================
# TAB 2: RESEARCH RESULTS
# ==========================================
with tab2:
    st.markdown("## 📊 Comparative Model Performance")
    
    # Section 1: Three-Model Comparison
    st.markdown("### 🔬 Final Model Comparison (RAF-DB Dataset)")
    
    model_comparison = pd.DataFrame({
        'Model': ['SVM (HOG+PCA)', 'CNN (MobileNetV2)', 'YOLO (Oversampled)'],
        'Accuracy': ['86%', '[Pending]', '88%'],
        'Precision': ['87%', '[Pending]', '88%'],
        'Recall': ['86%', '[Pending]', '88%'],
        'F1-Score': ['87%', '[Pending]', '88%'],
        'Inference Speed': ['~5-10 FPS', '[Pending]', '~60 FPS'],
        'Best Use Case': ['Edge devices', 'Batch processing', 'Real-time video']
    })
    
    st.dataframe(model_comparison, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # Section 2: YOLO Preprocessing Experiments
    st.markdown("### 🧪 YOLO Preprocessing Strategy Comparison")
    st.markdown("*Four preprocessing approaches tested (50 epochs each, 200 total training epochs)*")
    
    yolo_experiments = pd.DataFrame({
        'Experiment': [
            'Baseline',
            'Heavy Augmentation', 
            'Grayscale + Blur',
            'Oversampled (Selected)'
        ],
        'Accuracy': ['88%', '87%', '84%', '88%'],
        'Disgust Recall': ['60%', '40%', '40%', '66%'],
        'Fear Recall': ['47%', '47%', '47%', '59%'],
        'Key Finding': [
            'Strong baseline',
            'Over-regularization (-1%)',
            'Blur degraded features (-4%)',
            'Best minority class performance ✓'
        ]
    })
    
    st.dataframe(yolo_experiments, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # Section 3: Key Findings
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### ✅ What Worked")
        st.success("""
        **Data-Level Balancing (Oversampling)**
        - Improved disgust recall: 60% → 66% (+6%)
        - Improved fear recall: 47% → 59% (+12%)
        - Maintained 88% overall accuracy
        - No preprocessing artifacts
        """)
    
    with col2:
        st.markdown("### ❌ What Didn't Work")
        st.error("""
        **Gaussian Blur Preprocessing**
        - Accuracy dropped: 88% → 84% (-4%)
        - Removed micro-expressions YOLO needs
        - Same blur helped SVM (86% accuracy)
        - Lesson: Preprocessing must match model type
        """)
    
    st.markdown("---")
    
    # Section 4: Per-Class Performance (Selected Model)
    st.markdown("### 🎯 Per-Class Performance (Oversampled YOLO)")
    
    per_class = pd.DataFrame({
        'Emotion': ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise'],
        'Precision': ['84%', '73%', '77%', '96%', '84%', '83%', '88%'],
        'Recall': ['86%', '66%', '59%', '93%', '89%', '88%', '86%'],
        'F1-Score': ['85%', '70%', '67%', '94%', '86%', '85%', '87%'],
        'Test Samples': [162, 160, 74, 1185, 680, 478, 329]
    })
    
    st.dataframe(per_class, use_container_width=True, hide_index=True)
    
    st.info("""
    **Key Observations:**
    - ✅ Strong: Happy (93%), Neutral (89%), Sad (88%)
    - ⚠️ Challenging: Disgust (66%), Fear (59%)
    - 🔄 Main Confusion: Fear ↔ Surprise (22% confusion rate due to similar facial features)
    """)
    
    st.markdown("---")
    
    # Section 5: Research Contribution
    st.markdown("### 🔬 Research Contribution")
    
    st.markdown("""
    **Cross-Model Preprocessing Insight:**
    
    This study empirically demonstrated that preprocessing strategies must align with model architecture:
    
    | Model Type | Gaussian Blur Effect | Explanation |
    |------------|---------------------|-------------|
    | **SVM** (Manual features) | ✅ Improved to 86% | Smooths HOG gradients for consistency |
    | **YOLO** (Automatic learning) | ❌ Degraded to 84% | Removes learnable micro-features (-4%) |
    
    **Practical Implication:** Techniques effective for traditional ML can harm deep learning models. 
    YOLO achieved best results (88%) with minimal preprocessing (oversampling only) versus traditional 
    CV preprocessing (blur, grayscale, low resolution).
    """)
    
    st.markdown("---")
    
    # Section 6: Real-World Application Demo
    st.markdown("### 🚗 Real-World Application: Driver Monitoring")
    
    st.markdown("""
    This system demonstrates practical deployment for driver drowsiness detection:
    
    **Pipeline:**
    1. MediaPipe 3D Face Mesh detects face (±45° head tilt tolerance)
    2. YOLO classifies emotion in real-time (60 FPS on RTX 3060)
    3. Alert triggers if "tired" emotions detected for >5 seconds
    
    **Deployment Modes:**
    - 🎥 **Real-time:** Live webcam for continuous monitoring (driver safety, classroom engagement)
    - 📷 **Offline:** Image upload for batch analysis (interview footage, customer feedback)
    
    **Performance:**
    - Inference: ~7ms per face (YOLO classification)
    - Total latency: ~15ms (including face detection)
    - Throughput: 60+ FPS on RTX 3060 GPU
    """)

# --- FOOTER ---
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 10px;">
    <p><strong>BMCS2203 Artificial Intelligence Assignment | Group 4</strong></p>
    <p style="font-size: 12px;">Yee Zu Yao (SVM) | Daniel Lim (CNN) | Eizen Lim (YOLO) | Tutor: Mr. Tioh Keat Soon</p>
</div>
""", unsafe_allow_html=True)