"""
================================================
  SVM Real-Time Emotion Detection
  Balanced: 100 images per emotion class
  Run from Command Prompt:
      python emotion_svm.py
================================================
Requirements:
    pip install opencv-python scikit-learn scikit-image numpy matplotlib seaborn joblib
Dataset:
    Download FER2013 from https://www.kaggle.com/datasets/msambare/fer2013
    Extract so folder structure is:
        fer2013/
          train/
            angry/  disgust/  fear/  happy/  neutral/  sad/  surprise/
          test/
            angry/  disgust/  ...
"""

import os
import sys
import time
import cv2
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# ─────────────────────────────────────────────
#  CONFIGURATION  (edit these paths as needed)
# ─────────────────────────────────────────────
DATA_DIR       = 'fer2013/train'      # path to FER2013 train folder
MODEL_DIR      = 'models'            # where .pkl files are saved
IMAGES_PER_CLASS = 100               # balanced cap per emotion
IMG_SIZE       = (48, 48)
EMOTIONS       = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
USE_PCA        = True                # recommended: keeps 95% variance, trains faster
PCA_COMPONENTS = 150


# ─────────────────────────────────────────────
#  STEP 1 — LOAD BALANCED DATASET
# ─────────────────────────────────────────────
def load_balanced_dataset(data_dir, images_per_class=100):
    print("\n[1/5] Loading dataset (balanced: {} images per class)".format(images_per_class))
    X, y = [], []
    class_counts = {}

    for emotion in EMOTIONS:
        folder = os.path.join(data_dir, emotion)
        if not os.path.exists(folder):
            print(f"  WARNING: folder not found — {folder}")
            continue

        files = [f for f in os.listdir(folder)
                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        # Randomly sample exactly images_per_class (or all if fewer exist)
        np.random.seed(42)
        selected = np.random.choice(files, size=min(images_per_class, len(files)), replace=False)

        count = 0
        for fname in selected:
            img = cv2.imread(os.path.join(folder, fname), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, IMG_SIZE)
            X.append(img)
            y.append(emotion)
            count += 1

        class_counts[emotion] = count
        print(f"  {emotion:>10}: {count:>3} images loaded")

    print(f"\n  Total: {len(X)} images across {len(class_counts)} classes")
    return np.array(X), np.array(y), class_counts


# ─────────────────────────────────────────────
#  STEP 2 — HOG FEATURE EXTRACTION
# ─────────────────────────────────────────────
def extract_features(images):
    print("\n[2/5] Extracting HOG features...")
    features = []
    t0 = time.time()
    for i, img in enumerate(images):
        feat = hog(
            img,
            orientations=9,
            pixels_per_cell=(4, 4),
            cells_per_block=(2, 2),
            block_norm='L2-Hys'
        )
        features.append(feat)
        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            print(f"  {i+1}/{len(images)} done ({elapsed:.1f}s elapsed)")

    features = np.array(features)
    print(f"  Feature shape: {features.shape}  ({features.shape[1]} dims per image)")
    return features


# ─────────────────────────────────────────────
#  STEP 3 — PREPROCESS + OPTIONAL PCA
# ─────────────────────────────────────────────
def preprocess(X_train, X_test, use_pca=True, n_components=150):
    print("\n[3/5] Preprocessing (normalize + PCA)...")

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    pca = None
    if use_pca:
        pca = PCA(n_components=n_components, whiten=True, random_state=42)
        X_train_s = pca.fit_transform(X_train_s)
        X_test_s  = pca.transform(X_test_s)
        variance  = pca.explained_variance_ratio_.sum()
        print(f"  PCA: {n_components} components retain {variance:.1%} variance")
        print(f"  Feature dims reduced: {X_train.shape[1]} → {X_train_s.shape[1]}")
    else:
        print(f"  Skipping PCA — using raw {X_train_s.shape[1]} HOG features")

    return X_train_s, X_test_s, scaler, pca


# ─────────────────────────────────────────────
#  STEP 4 — TRAIN SVM
# ─────────────────────────────────────────────
def train_svm(X_train, y_train):
    print("\n[4/5] Training SVM (RBF kernel)...")
    print(f"  Training on {X_train.shape[0]} samples, {X_train.shape[1]} features")
    t0 = time.time()

    svm = SVC(
        kernel='rbf',
        C=1,
        gamma='scale',
        probability=True,
        cache_size=500,
        verbose=False
    )
    svm.fit(X_train, y_train)
    elapsed = time.time() - t0
    print(f"  Training complete in {elapsed:.1f}s")
    return svm


# ─────────────────────────────────────────────
#  STEP 5 — EVALUATE + SAVE
# ─────────────────────────────────────────────
def evaluate_and_save(svm, scaler, pca, le, X_test, y_test):
    os.makedirs(MODEL_DIR, exist_ok=True)
    print("\n[5/5] Evaluating model...")
    y_pred = svm.predict(X_test)
    y_true_labels = le.inverse_transform(y_test)
    y_pred_labels = le.inverse_transform(y_pred)

    print("\n" + "="*50)
    print(classification_report(y_true_labels, y_pred_labels, target_names=EMOTIONS))
    print("="*50)

    # Confusion matrix plot
    cm = confusion_matrix(y_true_labels, y_pred_labels, labels=EMOTIONS)
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt='d',
                xticklabels=EMOTIONS, yticklabels=EMOTIONS,
                cmap='Blues', linewidths=0.5)
    plt.title(f'SVM Emotion Confusion Matrix\n(balanced: {IMAGES_PER_CLASS} imgs/class)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, 'confusion_matrix.png'), dpi=150)
    print(f"\n  Confusion matrix saved → {MODEL_DIR}/confusion_matrix.png")
    plt.show()

    # Class distribution bar chart
    fig, ax = plt.subplots(figsize=(8, 4))
    counts = [(y_pred_labels == e).sum() for e in EMOTIONS]
    ax.bar(EMOTIONS, counts, color='steelblue', edgecolor='white')
    ax.set_title('Predicted emotion distribution on test set')
    ax.set_ylabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, 'prediction_distribution.png'), dpi=150)
    plt.show()

    # Save models
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(svm,    os.path.join(MODEL_DIR, 'emotion_svm.pkl'))
    joblib.dump(scaler, os.path.join(MODEL_DIR, 'emotion_scaler.pkl'))
    joblib.dump(le,     os.path.join(MODEL_DIR, 'emotion_le.pkl'))
    if pca:
        joblib.dump(pca, os.path.join(MODEL_DIR, 'emotion_pca.pkl'))

    print(f"\n  Models saved to ./{MODEL_DIR}/")
    print("    emotion_svm.pkl")
    print("    emotion_scaler.pkl")
    print("    emotion_le.pkl")
    if pca:
        print("    emotion_pca.pkl")


# ─────────────────────────────────────────────
#  REAL-TIME DETECTION (webcam)
# ─────────────────────────────────────────────
def run_realtime():
    print("\nLoading models from ./{}/ ...".format(MODEL_DIR))
    try:
        svm    = joblib.load(os.path.join(MODEL_DIR, 'emotion_svm.pkl'))
        scaler = joblib.load(os.path.join(MODEL_DIR, 'emotion_scaler.pkl'))
        le     = joblib.load(os.path.join(MODEL_DIR, 'emotion_le.pkl'))
        pca_path = os.path.join(MODEL_DIR, 'emotion_pca.pkl')
        pca    = joblib.load(pca_path) if os.path.exists(pca_path) else None
    except FileNotFoundError as e:
        print(f"  ERROR: {e}")
        print("  Run training first:  python emotion_svm.py --train")
        return

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("  ERROR: Cannot open webcam.")
        return

    print("  Webcam started — press Q to quit\n")

    EMOTION_COLORS = {
        'happy':   (0, 220, 100),
        'sad':     (200, 100, 50),
        'angry':   (0, 60, 220),
        'surprise':(0, 200, 220),
        'fear':    (180, 60, 220),
        'disgust': (60, 180, 60),
        'neutral': (180, 180, 180),
    }

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
        )

        for (x, y, w, h) in faces:
            roi  = cv2.resize(gray[y:y+h, x:x+w], IMG_SIZE)
            feat = hog(roi, orientations=9, pixels_per_cell=(4,4),
                       cells_per_block=(2,2), block_norm='L2-Hys').reshape(1,-1)
            feat = scaler.transform(feat)
            if pca:
                feat = pca.transform(feat)

            label  = le.inverse_transform(svm.predict(feat))[0]
            prob   = svm.predict_proba(feat).max()
            color  = EMOTION_COLORS.get(label, (200, 200, 200))

            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

            # Label background pill
            text    = f"{label}  {prob:.0%}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(frame, (x, y-th-14), (x+tw+10, y), color, -1)
            cv2.putText(frame, text, (x+5, y-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        # Status bar
        cv2.putText(frame, "SVM Emotion Detection  |  Q to quit",
                    (10, frame.shape[0]-12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180,180,180), 1)

        cv2.imshow('SVM Emotion Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("  Webcam closed.")


# ─────────────────────────────────────────────
#  MAIN ENTRY POINT
# ─────────────────────────────────────────────
def train_pipeline():
    total_start = time.time()

    # Check dataset exists
    if not os.path.exists(DATA_DIR):
        print(f"\nERROR: Dataset folder not found: {DATA_DIR}")
        print("Download FER2013 from https://www.kaggle.com/datasets/msambare/fer2013")
        print("Extract so that fer2013/train/angry/, fer2013/train/happy/ etc. exist.")
        sys.exit(1)

    # Load
    images, labels, class_counts = load_balanced_dataset(DATA_DIR, IMAGES_PER_CLASS)

    # Encode labels
    le    = LabelEncoder()
    y_enc = le.fit_transform(labels)

    # Extract HOG features
    X = extract_features(images)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    # Preprocess
    X_train_p, X_test_p, scaler, pca = preprocess(X_train, X_test, USE_PCA, PCA_COMPONENTS)

    # Train
    svm = train_svm(X_train_p, y_train)

    # Evaluate + save
    evaluate_and_save(svm, scaler, pca, le, X_test_p, y_test)

    total = time.time() - total_start
    mins, secs = divmod(int(total), 60)
    print(f"\nTotal time: {mins}m {secs}s")
    print("\nDone! Run detection with:  python emotion_svm.py --detect")


if __name__ == '__main__':
    if '--detect' in sys.argv:
        run_realtime()
    elif '--train' in sys.argv:
        train_pipeline()
    else:
        print("\nUsage:")
        print("  python emotion_svm.py --train     Train the model")
        print("  python emotion_svm.py --detect    Run real-time webcam detection")
        print("\nStarting training by default...\n")
        train_pipeline()