import os
import cv2
import joblib
import numpy as np
from skimage.feature import hog, local_binary_pattern
from collections import deque
from datetime import datetime
import warnings

# Suppress sklearn warnings about feature names if they appear
warnings.filterwarnings("ignore", category=UserWarning)

# ─────────────────────────────────────────────
#  1. SETUP & LOAD BUNDLE
# ─────────────────────────────────────────────
MODEL_PATH = r'C:\Users\User\Desktop\AiAssignment\Ai-Assignment\emotion_model_final.pkl'

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model bundle not found at {MODEL_PATH}")

print("Loading trained model bundle...")
bundle = joblib.load(MODEL_PATH)

model  = bundle['model']
scaler = bundle['scaler']
pca    = bundle['pca']
le     = bundle['le']

print("✅ Loaded 7 emotion classes:", list(le.classes_))

IMG_SIZE = (200, 200)

# Create folder to save captures
CAPTURE_DIR = "captures"
os.makedirs(CAPTURE_DIR, exist_ok=True)

# ─────────────────────────────────────────────
#  2. FEATURE EXTRACTION
# ─────────────────────────────────────────────
def extract_detailed_features(img_gray):
    # 1. HOG features
    h_feat = hog(img_gray, orientations=9, pixels_per_cell=(8, 8),
                 cells_per_block=(2, 2), block_norm='L2-Hys')
    
    # 2. LBP - Uniform LBP with P=8 produces 10 unique uniform patterns
    lbp = local_binary_pattern(img_gray, P=8, R=1, method='uniform')
    
    # Bins set to 10 to perfectly match the 20746 total features
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=10, range=(0, 10), density=True)
    
    return np.concatenate([h_feat, lbp_hist])

# ─────────────────────────────────────────────
#  3. DETECTION ENGINE
# ─────────────────────────────────────────────
def start_detection():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("\n✅ System Online!")
    print("   Press 'Q' to quit")
    print("   Press 'C' to capture current face\n")

    recent_probas = deque(maxlen=15)
    frame_count = 0
    last_label = "NEUTRAL"

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            display_frame = frame.copy()

            if frame_count % 2 == 0:   # process every other frame for speed
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(50, 50))

                for (x, y, w, h) in faces:
                    # Crop face
                    roi_gray = gray[y:y+h, x:x+w]
                    
                    # FIX 1: Histogram Equalization to normalize lighting/contrast
                    # This prevents ambient webcam lighting from skewing the model
                    roi_gray = cv2.equalizeHist(roi_gray)
                    
                    # FIX 2: Resize
                    roi_resized = cv2.resize(roi_gray, IMG_SIZE)
                    
                    # Extract features and format for the model
                    features = extract_detailed_features(roi_resized).reshape(1, -1)
                    
                    # Predict pipeline
                    features_scaled = scaler.transform(features)
                    features_pca = pca.transform(features_scaled)
                    proba = model.predict_proba(features_pca)[0]
                    
                    recent_probas.append(proba)
                    
                    # Smoothed prediction logic
                    if len(recent_probas) >= 5:
                        avg_proba = np.mean(list(recent_probas), axis=0)
                        max_idx = np.argmax(avg_proba)
                        smoothed_label = le.inverse_transform([max_idx])[0]
                        smoothed_prob = avg_proba[max_idx]
                    else:
                        smoothed_label = le.inverse_transform([np.argmax(proba)])[0]
                        smoothed_prob = proba.max()

                    # Anti-flash logic 
                    if smoothed_prob > 0.35 or smoothed_label == last_label:
                        last_label = smoothed_label
                    else:
                        smoothed_label = last_label

                    # Draw on screen
                    color = (0, 255, 0)
                    cv2.rectangle(display_frame, (x, y), (x+w, y+h), color, 2)
                    text = f"{smoothed_label.upper()} ({smoothed_prob:.1%})"
                    cv2.putText(display_frame, text, (x, y-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            # Capture key handling
            key = cv2.waitKey(1) & 0xFF
            if key == ord('c') or key == ord('C'):
                if 'faces' in locals() and len(faces) > 0:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    face_crop = frame[y:y+h, x:x+w]
                    
                    img_path = os.path.join(CAPTURE_DIR, f"face_{timestamp}.jpg")
                    cv2.imwrite(img_path, face_crop)
                    
                    prob_path = os.path.join(CAPTURE_DIR, f"face_{timestamp}.txt")
                    with open(prob_path, "w") as f:
                        f.write(f"Label: {smoothed_label}\n")
                        f.write(f"Confidence: {smoothed_prob:.1%}\n\n")
                        f.write("Raw probabilities:\n")
                        for label_name, p in zip(le.classes_, proba):
                            f.write(f"  {label_name:8} : {p:.4f}\n")
                    
                    print(f"✅ Captured! → {img_path}")

            cv2.imshow('Emotion Detection System', display_frame)
            if key == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    start_detection()