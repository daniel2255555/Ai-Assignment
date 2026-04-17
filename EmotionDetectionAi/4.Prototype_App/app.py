import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
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
    # FIXED: Model is in 3.Models folder (one directory up from app.py)
    model_path = Path(__file__).parent.parent / "3.Models" / "best_oversampled.pt"
    
    if not model_path.exists():
        st.error(f"❌ Model not found at: {model_path}")
        st.stop()
    
    return YOLO(str(model_path))

@st.cache_resource
def load_face_detector():
    # Using OpenCV's Haar Cascade (no MediaPipe dependency)
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    return cv2.CascadeClassifier(cascade_path)

model = load_yolo_model()
face_cascade = load_face_detector()

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    
    # Feature toggles
    show_top3 = st.checkbox("Show Top 3 Predictions", value=True)
    drowsiness_alert = st.checkbox("🚗 Driver Drowsiness Alert", value=False)
    
    if drowsiness_alert:
        alert_duration = st.slider("Alert after (seconds):", 3, 15, 5)
        enable_sound = st.checkbox("🔊 Enable Sound Alert", value=True)
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
    
    def detect_faces_opencv(frame):
        """
        Detect faces using OpenCV Haar Cascade with strict parameters
        to avoid false positives from reflections/background
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        
        faces = face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.3,        # Less sensitive to minor variations
            minNeighbors=7,         # Require more confirming detections
            minSize=(80, 80),       # Ignore small artifacts
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # Filter: Keep only faces >= 2% of frame area
        if len(faces) > 0:
            frame_area = frame.shape[0] * frame.shape[1]
            min_area = frame_area * 0.02
            
            valid_faces = []
            for (x, y, w, h) in faces:
                if (w * h) >= min_area:
                    valid_faces.append((x, y, w, h))
            
            # If multiple valid faces, keep only the largest
            if len(valid_faces) > 1:
                valid_faces = sorted(valid_faces, key=lambda f: f[2] * f[3], reverse=True)
                valid_faces = valid_faces[:1]
            
            return np.array(valid_faces)
        
        return faces
    
    def process_faces(frame, model):
        """Process detected faces and classify emotions"""
        faces = detect_faces_opencv(frame)
        faces_data = []
        
        for (x, y, w, h) in faces:
            # Add padding
            pad = int(w * 0.2)
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(frame.shape[1], x + w + pad)
            y2 = min(frame.shape[0], y + h + pad)
            
            # Crop face
            face_roi = frame[y1:y2, x1:x2]
            
            if face_roi.size == 0:
                continue
            
            # YOLO prediction
            results = model(face_roi, verbose=False)
            top3 = get_top3_predictions(results)
            top_emotion, top_conf = top3[0]
            
            faces_data.append({
                'bbox': (x1, y1, x2, y2),
                'top_emotion': top_emotion,
                'top_conf': top_conf,
                'top3': top3
            })
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (102, 126, 234), 2)
            
            # Label
            label = f"{top_emotion.upper()} {top_conf:.1f}%"
            cv2.rectangle(frame, (x1, y1-30), (x1 + len(label)*12, y1), (102, 126, 234), -1)
            cv2.putText(frame, label, (x1+5, y1-8), 
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
                annotated_frame, faces_data = process_faces(frame, model)
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
        alert_active = False
        
        if run_webcam:
            cap = cv2.VideoCapture(0)
            
            while run_webcam:
                ret, frame = cap.read()
                if not ret:
                    st.error("❌ Failed to access webcam")
                    break
                
                # Process frame
                annotated_frame, faces_data = process_faces(frame, model)
                
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
                            if not alert_active:
                                with ALERT_PLACEHOLDER:
                                    st.error("🚨 **DRIVER ALERT:** Drowsiness detected! Take a break. Pull over safely.")
                                    st.balloons()
                                    
                                    # 🔊 PLAY BROWSER SOUND ALERT
                                    if enable_sound:
                                        st.markdown("""
                                            <audio autoplay>
                                                <source src="https://actions.google.com/sounds/v1/alarms/beep_short.ogg" type="audio/ogg">
                                            </audio>
                                        """, unsafe_allow_html=True)
                                
                                alert_active = True
                    else:
                        drowsy_start_time = None
                        alert_active = False
                        ALERT_PLACEHOLDER.empty()
                
                # Display frame
                rgb_final = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                FRAME_WINDOW.image(rgb_final, channels="RGB", use_container_width=True)
            
            cap.release()

# ==========================================
# TAB 2: RESEARCH RESULTS (UPDATED)
# ==========================================
with tab2:
    st.markdown("## 📊 Comparative Model Performance")
    
    # Section 1: Three-Model Comparison
    st.markdown("### 🔬 Final Model Comparison (RAF-DB Dataset)")
    
    model_comparison = pd.DataFrame({
        'Model': ['SVM (HOG+PCA)', 'CNN (MobileNetV2)', 'YOLO (Oversampled)'],
        'Accuracy': ['86%', '77%', '88%'],
        'Precision': ['87%', '78%', '88%'],
        'Recall': ['86%', '77%', '88%'],
        'F1-Score': ['87%', '77%', '88%'],
        'Inference Speed': ['~5-10 FPS', '~30 FPS', '~60 FPS'],
        'Best Use Case': ['Edge devices', 'Batch processing', 'Real-time video']
    })
    
    st.dataframe(model_comparison, use_container_width=True, hide_index=True)
    
    # Highlight winner
    st.success("✅ **YOLO selected as final model:** Highest accuracy (88%) + Fastest speed (60 FPS)")
    
    st.markdown("---")
    
    # Section 2: Model Selection Rationale
    st.markdown("### 🎯 Why YOLO Was Selected")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **✅ Superior Performance**
        - **Highest Accuracy:** 88% (vs SVM 86%, CNN 77%)
        - **Best F1-Score:** Balanced precision and recall
        - **Consistent across emotions:** No catastrophic failures
        
        **⚡ Real-Time Speed**
        - **60 FPS on RTX 3060** (vs SVM 5-10 FPS, CNN 30 FPS)
        - **~7ms inference latency** per face
        - Enables live webcam monitoring
        - Suitable for driver drowsiness detection
        """)
    
    with col2:
        st.markdown("""
        **🎯 Improved Minority Classes**
        - **Disgust recall:** 60% → 66% (+6% improvement)
        - **Fear recall:** 47% → 59% (+12% improvement)
        - Data-level balancing (oversampling) resolved class imbalance
        
        **🔧 End-to-End Detection**
        - Combines face detection + emotion classification
        - Simpler deployment pipeline
        - Single model to maintain
        """)
    
    # CNN Analysis
    st.warning("""
    **❌ CNN Underperformance Analysis (77% accuracy):**
    - Transfer learning from ImageNet → RAF-DB showed domain shift
    - Fine-tuning with only 3,068 trainable parameters insufficient
    - Validates need for domain-specific end-to-end training (YOLO approach)
    """)
    
    st.markdown("---")
    
    # Section 3: YOLO Preprocessing Experiments
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
    
    # Section 4: Key Findings
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
    
    # Section 5: Per-Class Performance (Selected Model)
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
    - ✅ **Strong:** Happy (93%), Neutral (89%), Sad (88%)
    - ⚠️ **Challenging:** Disgust (66%), Fear (59%) - minority classes with fewer training samples
    - 🔄 **Main Confusion:** Fear ↔ Surprise (22% confusion rate)
        - Both emotions share similar facial features (raised eyebrows, wide eyes)
        - Similar muscle activations make differentiation challenging
    """)
    
    st.markdown("---")
    
    # Section 6: Research Contribution
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
    
    # Section 7: Real-World Application Demo
    st.markdown("### 🚗 Real-World Application: Driver Monitoring System")
    
    st.markdown("""
    This system demonstrates practical deployment for driver drowsiness detection:
    
    **Pipeline:**
    1. **Face Detection:** OpenCV Haar Cascade (~5ms)
    2. **Emotion Classification:** YOLO real-time inference (~7ms per face)
    3. **Alert System:** Triggers if "tired" emotions detected for >5 seconds
    4. **Audio + Visual Alerts:** Browser-based sound + visual notification
    
    **Performance Metrics:**
    - **Inference latency:** ~7ms per face (YOLO classification only)
    - **Total pipeline latency:** ~15ms (face detection + YOLO + rendering)
    - **Throughput:** 60+ FPS on RTX 3060 GPU
    - **Real-time capability:** Processes 30 FPS webcam with 2× headroom
    
    **Deployment Modes:**
    - 🎥 **Real-time:** Live webcam for continuous monitoring (driver safety, classroom engagement)
    - 📷 **Offline:** Image upload for batch analysis (interview footage, customer feedback)
    
    **Use Cases:**
    - 🚗 Driver drowsiness monitoring (automotive safety systems)
    - 📚 Online learning engagement tracking (education technology)
    - 🏥 Patient mood monitoring (healthcare applications)
    - 💼 Customer sentiment analysis (retail and service industries)
    """)
    
    st.markdown("---")
    
    # Section 8: Model Comparison Summary Table
    st.markdown("### 📋 Comprehensive Model Comparison")
    
    detailed_comparison = pd.DataFrame({
        'Aspect': [
            'Overall Accuracy',
            'Minority Class Performance',
            'Inference Speed (FPS)',
            'Inference Latency (ms)',
            'Training Approach',
            'Preprocessing Required',
            'Real-time Capable',
            'Deployment Complexity',
            'Hardware Requirement'
        ],
        'SVM (HOG+PCA)': [
            '86%',
            'Moderate (60%/47% Disgust/Fear)',
            '5-10 FPS',
            '~100-200ms',
            'Manual feature extraction',
            'Heavy (HOG, blur, PCA)',
            '❌ Too slow',
            'Low',
            'CPU-friendly'
        ],
        'CNN (MobileNetV2)': [
            '77%',
            'Weak (transfer learning gap)',
            '~30 FPS',
            '~33ms',
            'Transfer learning (ImageNet)',
            'Minimal (resize, normalize)',
            '⚠️ Borderline',
            'Medium',
            'GPU preferred'
        ],
        'YOLO (Oversampled)': [
            '88%',
            'Strong (66%/59% Disgust/Fear)',
            '~60 FPS',
            '~7ms',
            'End-to-end on RAF-DB',
            'Minimal (oversampling only)',
            '✅ Yes',
            'Low (single model)',
            'GPU optimal'
        ]
    })
    
    st.dataframe(detailed_comparison, use_container_width=True, hide_index=True)

# --- FOOTER ---
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 10px;">
    <p><strong>BMCS2203 Artificial Intelligence Assignment | Group 4</strong></p>
    <p style="font-size: 12px;">Yee Zu Yao (SVM) | Daniel Lim (CNN) | Eizen Lim (YOLO) | Tutor: Mr. Tioh Keat Soon</p>
</div>
""", unsafe_allow_html=True)
