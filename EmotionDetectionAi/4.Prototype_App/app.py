import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import mediapipe as mp

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Emotion AI Classifier", layout="wide", page_icon="🎭")

st.markdown("""
    <style>
    .main-banner {
        background: linear-gradient(90deg, #6a5acd 0%, #8a2be2 100%);
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        color: white;
        margin-bottom: 20px;
    }
    </style>
    <div class="main-banner">
        <h1>🎭 Real-Time Emotion Classifier</h1>
        <p>Production Model: YOLOv8 (Oversampled)</p>
    </div>
""", unsafe_allow_html=True)

# --- LOAD MODELS ---
@st.cache_resource
def load_yolo_model():
    return YOLO(r"C:\Users\clogg\Desktop\Yolo_Master_folder\3.Models\best_oversampled.pt")

@st.cache_resource
def load_face_detector():
    # UPGRADED: Using 3D Face Mesh for extreme occlusion and tilt resistance
    mp_face_mesh = mp.solutions.face_mesh
    return mp_face_mesh.FaceMesh(max_num_faces=5, min_detection_confidence=0.5, min_tracking_confidence=0.5)

model = load_yolo_model()
face_mesh = load_face_detector()

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ System Status")
    st.success("YOLOv8 Classifier Loaded")
    st.success("MediaPipe 3D Mesh Loaded")
    
    st.markdown("---")
    st.markdown("### 🚀 Pipeline Features")
    st.markdown("""
    - **Two-Stage Processing:** Isolates Region of Interest (ROI) before classification.
    - **Extreme Robustness:** Uses 468-point 3D Face Mesh to maintain tracking through severe head tilts and hand occlusions.
    - **Class Balancing:** Trained on an oversampled dataset to ensure high recall for rare emotions.
    """)

# --- MAIN CONTENT AREA ---
st.subheader("📂 Input Selection")
input_mode = st.radio("Select Input Mode:", ["Image Upload", "Live Webcam"], horizontal=True)
st.markdown("---")

def process_mesh_rois(frame, results_mesh, model):
    """Extracts bounding box from 3D mesh points for extreme tilt tracking"""
    if results_mesh.multi_face_landmarks:
        h, w, _ = frame.shape
        for face_landmarks in results_mesh.multi_face_landmarks:
            
            # Find the absolute min/max X and Y from all 468 facial landmarks
            x_coords = [landmark.x * w for landmark in face_landmarks.landmark]
            y_coords = [landmark.y * h for landmark in face_landmarks.landmark]
            
            x_min, x_max = int(min(x_coords)), int(max(x_coords))
            y_min, y_max = int(min(y_coords)), int(max(y_coords))
            
            box_w = x_max - x_min
            box_h = y_max - y_min
            
            # Pad the bounding box slightly to capture hair/chin
            pad_x, pad_y = int(box_w * 0.15), int(box_h * 0.15)
            x = max(0, x_min - pad_x)
            y = max(0, y_min - pad_y)
            x_end = min(w, x_max + pad_x)
            y_end = min(h, y_max + pad_y)
            
            # Crop the face
            face_roi = frame[y:y_end, x:x_end]
            
            if face_roi.size == 0:
                continue

            # YOLO Inference
            results = model(face_roi, verbose=False)
            top_class_id = results[0].probs.top1
            emotion = results[0].names[top_class_id]
            confidence = results[0].probs.top1conf.item() * 100
            
            # Draw UI
            cv2.rectangle(frame, (x, y), (x_end, y_end), (138, 43, 226), 3)
            label = f"{emotion.upper()} {confidence:.1f}%"
            cv2.rectangle(frame, (x, y-35), (x + len(label)*15, y), (138, 43, 226), -1)
            cv2.putText(frame, label, (x+5, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
    return frame

# ==========================================
# MODE 1: IMAGE UPLOAD
# ==========================================
if input_mode == "Image Upload":
    st.subheader("Upload an Image")
    uploaded_file = st.file_uploader("Drag and drop file here (JPG, PNG, JPEG)", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        with st.spinner('Tracking facial mesh and analyzing emotions...'):
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mesh_results = face_mesh.process(frame_rgb)
            
            if not mesh_results.multi_face_landmarks:
                st.warning("No faces detected.")
                st.image(image, caption="Original Image", use_column_width=True)
            else:
                annotated_frame = process_mesh_rois(frame, mesh_results, model)
                rgb_final_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                st.image(rgb_final_frame, caption="Analysis Complete", use_column_width=True)

# ==========================================
# MODE 2: LIVE WEBCAM
# ==========================================
elif input_mode == "Live Webcam":
    st.subheader("Live Webcam Analysis")
    run_webcam = st.checkbox("Start Webcam")
    FRAME_WINDOW = st.empty()
    
    if run_webcam:
        cap = cv2.VideoCapture(0)
        while run_webcam:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Process Mesh
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mesh_results = face_mesh.process(frame_rgb)
            
            # Draw Output
            annotated_frame = process_mesh_rois(frame, mesh_results, model)
            
            # Push to UI
            rgb_final_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(rgb_final_frame, channels="RGB", use_column_width=True)
                
        cap.release()