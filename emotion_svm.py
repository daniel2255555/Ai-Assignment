"""
================================================
  SVM Emotion Detection — RAF-DB Dataset
  Traditional Machine Learning (HOG + LBP + PCA + SVM)
  BMCS2203 · Image Processing & Computer Vision

  Run from Command Prompt:
      python emotion_svm_rafdb.py --install   (first time only)
      python emotion_svm_rafdb.py --train
      python emotion_svm_rafdb.py --detect
      python emotion_svm_rafdb.py --evaluate

  Folder structure required:
      Ai-Assignment/
      ├── emotion_svm.py   ← this file
      ├── RAF-DB/
      │   └── train/
      │       ├── angry/
      │       ├── disgust/
      │       ├── fear/
      │       ├── happy/
      │       ├── neutral/
      │       ├── sad/
      │       └── surprise/
      └── models/               ← auto-created on --train
================================================
"""

import os
import sys
import time
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
#  CONFIGURATION  — edit these if needed
# ─────────────────────────────────────────────
DATA_DIR  = r'RAF-DB\train'       # relative to this script's folder
MODEL_DIR = 'models'
IMG_SIZE  = (100, 100)

# BUG FIX: 'suprise' → 'surprise' (typo in original)
EMOTIONS  = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Limit images per class to prevent RAM crash on large datasets
# RAF-DB has ~3k-12k per class — 1500 is a safe balanced cap
IMAGES_PER_CLASS = 4772

# SVM params — tuned for RAF-DB
SVM_C     = 10
SVM_GAMMA = 'scale'


# ─────────────────────────────────────────────
#  STEP 0 — Auto-install missing packages
# ─────────────────────────────────────────────
def install_dependencies():
    import subprocess
    packages = [
        'opencv-python',
        'scikit-learn',
        'scikit-image',
        'imbalanced-learn',
        'numpy',
        'matplotlib',
        'seaborn',
        'joblib',
    ]
    print("\n[INSTALL] Installing required packages...")
    for pkg in packages:
        print(f"  pip install {pkg}")
        subprocess.run(
            [sys.executable, '-m', 'pip', 'install', pkg, '-q'],
            check=False
        )
    print("[INSTALL] Done. Now run:  python emotion_svm_rafdb.py --train\n")


# ─────────────────────────────────────────────
#  STEP 1 — Feature extraction: HOG + LBP
# ─────────────────────────────────────────────
def extract_features(img):
    """
    Combined HOG + LBP feature vector.

    HOG  → captures gradient orientation (face shape, contours, edges)
    LBP  → captures local texture (skin, wrinkles, micro-patterns)
    Both → gives SVM richer discriminative features vs HOG alone (+3-6% accuracy)

    BUG FIX from original:
      - visualize=False  (was missing → hog() returned tuple → shape crash)
      - bins=np.arange(0,11) (was range=(0,10) → missing last bin edge)
    """
    from skimage.feature import hog, local_binary_pattern
    import numpy as np

    # HOG features — shape/gradient
    hog_feat = hog(
        img,
        orientations=9,
        pixels_per_cell=(4, 4),
        cells_per_block=(2, 2),
        block_norm='L2-Hys',
        visualize=False          # ← CRITICAL FIX: False not True
    )

    # LBP features — texture
    lbp = local_binary_pattern(img, P=8, R=1, method='uniform')
    lbp_hist, _ = np.histogram(
        lbp.ravel(),
        bins=np.arange(0, 11),  # ← CRITICAL FIX: arange not range=(0,10)
        density=True
    )

    return np.concatenate([hog_feat, lbp_hist])


# ─────────────────────────────────────────────
#  STEP 2 — Load RAF-DB dataset
# ─────────────────────────────────────────────
def load_dataset():
    import cv2
    import numpy as np

    # Resolve DATA_DIR relative to this script's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path  = os.path.join(script_dir, DATA_DIR)

    if not os.path.exists(data_path):
        print(f"\nERROR: Dataset not found at: {data_path}")
        print("Expected structure: RAF-DB/train/angry/  happy/  sad/ ...")
        sys.exit(1)

    print(f"\n[1/5] Loading RAF-DB from: {data_path}")
    print(f"      Mode: {IMAGES_PER_CLASS} images per class (balanced)\n")

    X, y = [], []
    class_counts = {}

    for emotion in EMOTIONS:
        folder = os.path.join(data_path, emotion)

        if not os.path.exists(folder):
            print(f"  WARNING: Missing folder — {folder}")
            continue

        files = [
            f for f in os.listdir(folder)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]

        # Balanced random sampling
        np.random.seed(42)
        if IMAGES_PER_CLASS and len(files) > IMAGES_PER_CLASS:
            files = list(np.random.choice(files, IMAGES_PER_CLASS, replace=False))

        count = 0
        for fname in files:
            img = cv2.imread(os.path.join(folder, fname), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, IMG_SIZE)
            X.append(extract_features(img))
            y.append(emotion)
            count += 1

        class_counts[emotion] = count
        print(f"  {emotion:>10}: {count:>4} images")

    X = np.array(X)
    y = np.array(y)
    print(f"\n  Total: {len(X)} images | Feature dims: {X.shape[1]}")
    return X, y


# ─────────────────────────────────────────────
#  STEP 3 — Preprocess: scale + PCA + SMOTE
# ─────────────────────────────────────────────
def preprocess(X_train, y_train, X_test):
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    print("\n[2/5] Preprocessing: StandardScaler → PCA → SMOTE...")

    # StandardScaler — critical for SVM (sensitive to feature scale)
    scaler    = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)   # use train stats on test

    # PCA — keeps 95% variance, reduces dims for faster SVM
    pca       = PCA(n_components=0.95, whiten=True, random_state=42)
    X_train_p = pca.fit_transform(X_train_s)
    X_test_p  = pca.transform(X_test_s)

    print(f"  PCA: {pca.n_components_} components retain "
          f"{pca.explained_variance_ratio_.sum():.1%} variance")
    print(f"  Dims: {X_train.shape[1]} → {X_train_p.shape[1]}")

    # SMOTE — synthesises new samples for minority classes
    # Fixes class imbalance in RAF-DB (disgust/fear are underrepresented)
    # BUG FIX: apply SMOTE AFTER PCA, not before (less RAM, faster)
    try:
        from imblearn.over_sampling import SMOTE
        unique, counts = np.unique(y_train, return_counts=True)
        print(f"\n  Class counts before SMOTE: {dict(zip(unique, counts))}")
        smote    = SMOTE(random_state=42, k_neighbors=3)
        X_bal, y_bal = smote.fit_resample(X_train_p, y_train)
        unique2, counts2 = np.unique(y_bal, return_counts=True)
        print(f"  Class counts after  SMOTE: {dict(zip(unique2, counts2))}")
    except ImportError:
        print("  SMOTE not available — run --install first. Skipping.")
        X_bal, y_bal = X_train_p, y_train

    return X_bal, y_bal, X_test_p, scaler, pca


# ─────────────────────────────────────────────
#  STEP 4 — Train RBF SVM
# ─────────────────────────────────────────────
def train_svm(X_train, y_train):
    from sklearn.svm import SVC

    print(f"\n[3/5] Training RBF SVM (C={SVM_C}, gamma={SVM_GAMMA})...")
    print(f"  Samples: {X_train.shape[0]} | Features: {X_train.shape[1]}")

    t0  = time.time()
    svm = SVC(
        kernel='rbf',
        C=SVM_C,
        gamma=SVM_GAMMA,
        probability=True,
        class_weight='balanced',  # handles remaining imbalance
        cache_size=1000,
        random_state=42
    )
    svm.fit(X_train, y_train)

    elapsed = time.time() - t0
    mins, secs = divmod(int(elapsed), 60)
    print(f"  Training done in {mins}m {secs}s")
    return svm


# ─────────────────────────────────────────────
#  STEP 5 — Evaluate + save plots
# ─────────────────────────────────────────────
def evaluate_and_save(svm, scaler, pca, le, X_test, y_test):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')   # no display needed — saves to file
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import (
        classification_report, confusion_matrix,
        accuracy_score, precision_score, recall_score, f1_score
    )

    os.makedirs(MODEL_DIR, exist_ok=True)
    print("\n[4/5] Evaluating model...")

    y_pred     = svm.predict(X_test)
    y_true_lbl = le.inverse_transform(y_test)
    y_pred_lbl = le.inverse_transform(y_pred)

    acc  = accuracy_score (y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec  = recall_score   (y_test, y_pred, average='weighted', zero_division=0)
    f1   = f1_score       (y_test, y_pred, average='weighted', zero_division=0)

    print("\n" + "=" * 55)
    print(f"  Accuracy  : {acc:.1%}")
    print(f"  Precision : {prec:.1%}  (weighted)")
    print(f"  Recall    : {rec:.1%}  (weighted)")
    print(f"  F1-Score  : {f1:.1%}  (weighted)")
    print("=" * 55)
    print()
    print(classification_report(y_true_lbl, y_pred_lbl, target_names=EMOTIONS))

    # ── Confusion matrix ──────────────────────────────────────
    cm = confusion_matrix(y_true_lbl, y_pred_lbl, labels=EMOTIONS)
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    sns.heatmap(
        cm, annot=True, fmt='d',
        xticklabels=EMOTIONS, yticklabels=EMOTIONS,
        cmap='Blues', linewidths=0.5, ax=axes[0]
    )
    axes[0].set_title(
        f'SVM Emotion Confusion Matrix (RAF-DB)\n'
        f'Acc: {acc:.1%}  |  Prec: {prec:.1%}  |  F1: {f1:.1%}',
        fontsize=12
    )
    axes[0].set_ylabel('True Label')
    axes[0].set_xlabel('Predicted Label')
    axes[0].tick_params(axis='x', rotation=30)

    # ── Per-class accuracy bar chart ──────────────────────────
    per_class_acc, per_class_n = [], []
    for e in EMOTIONS:
        mask = y_true_lbl == e
        n    = mask.sum()
        a    = (y_pred_lbl[mask] == e).mean() if n > 0 else 0
        per_class_acc.append(a)
        per_class_n.append(n)

    colors = [
        '#2ecc71' if a >= 0.70 else
        '#f39c12' if a >= 0.50 else
        '#e74c3c'
        for a in per_class_acc
    ]
    bars = axes[1].bar(
        EMOTIONS, [a * 100 for a in per_class_acc],
        color=colors, edgecolor='white'
    )
    axes[1].axhline(acc * 100, color='navy', linestyle='--',
                    label=f'Overall {acc:.1%}')
    axes[1].set_title('Per-class accuracy (%)', fontsize=12)
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_ylim(0, 108)
    axes[1].tick_params(axis='x', rotation=30)
    axes[1].legend()

    for bar, n, a in zip(bars, per_class_n, per_class_acc):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f'{a:.0%}\n(n={n})',
            ha='center', va='bottom', fontsize=8
        )

    plt.tight_layout()
    cm_path = os.path.join(MODEL_DIR, 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Confusion matrix saved → {cm_path}")

    # ── Learning curve (safe: cv=2, n_jobs=1) ─────────────────
    print("\n  Generating learning curve (cv=2, safe mode)...")
    from sklearn.model_selection import learning_curve
    import numpy as np

    # Use a subset for speed — learning curve on full data is very slow
    n_samples = min(3000, len(X_test) * 5)
    train_sizes, train_scores, val_scores = learning_curve(
        svm, X_test, y_test,
        cv=2, n_jobs=1,
        train_sizes=np.linspace(0.3, 1.0, 4),
        scoring='accuracy'
    )

    fig2, ax = plt.subplots(figsize=(9, 5))
    ax.plot(train_sizes, np.mean(train_scores, axis=1), 'o-',
            color='steelblue', label='Training accuracy')
    ax.plot(train_sizes, np.mean(val_scores,   axis=1), 'o-',
            color='coral',     label='Validation accuracy')
    ax.fill_between(train_sizes,
                    np.mean(train_scores, axis=1) - np.std(train_scores, axis=1),
                    np.mean(train_scores, axis=1) + np.std(train_scores, axis=1),
                    alpha=0.15, color='steelblue')
    ax.fill_between(train_sizes,
                    np.mean(val_scores, axis=1) - np.std(val_scores, axis=1),
                    np.mean(val_scores, axis=1) + np.std(val_scores, axis=1),
                    alpha=0.15, color='coral')
    ax.set_title('SVM Learning Curve — RAF-DB\n(HOG + LBP + PCA + SMOTE + RBF)')
    ax.set_xlabel('Training samples')
    ax.set_ylabel('Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    lc_path = os.path.join(MODEL_DIR, 'learning_curve.png')
    plt.savefig(lc_path, dpi=150)
    plt.close()
    print(f"  Learning curve saved  → {lc_path}")

    # ── Save metrics summary ──────────────────────────────────
    with open(os.path.join(MODEL_DIR, 'metrics.txt'), 'w') as f:
        f.write(f"RAF-DB SVM Results\n")
        f.write(f"==================\n")
        f.write(f"Accuracy  : {acc:.4f} ({acc:.1%})\n")
        f.write(f"Precision : {prec:.4f} ({prec:.1%})\n")
        f.write(f"Recall    : {rec:.4f} ({rec:.1%})\n")
        f.write(f"F1-Score  : {f1:.4f} ({f1:.1%})\n\n")
        f.write(classification_report(y_true_lbl, y_pred_lbl, target_names=EMOTIONS))
    print(f"  Metrics saved         → {MODEL_DIR}/metrics.txt")

    return acc, prec, rec, f1


# ─────────────────────────────────────────────
#  STEP 6 — Save models
# ─────────────────────────────────────────────
def save_models(svm, scaler, pca, le):
    import joblib
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("\n[5/5] Saving models...")
    bundle = {
        'svm'    : svm,
        'scaler' : scaler,
        'pca'    : pca,
        'le'     : le,
        'emotions': EMOTIONS,
        'img_size': IMG_SIZE,
    }
    path = os.path.join(MODEL_DIR, 'emotion_model.pkl')
    joblib.dump(bundle, path)
    size_mb = os.path.getsize(path) / 1024 / 1024
    print(f"  Saved: {path}  ({size_mb:.1f} MB)")
    print(f"\nDone! Run detection with:")
    print(f"  python emotion_svm_rafdb.py --detect")


# ─────────────────────────────────────────────
#  TRAIN PIPELINE
# ─────────────────────────────────────────────
def train_pipeline():
    import joblib
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split

    total_start = time.time()

    # Load
    X, y = load_dataset()

    # Encode
    le    = LabelEncoder()
    y_enc = le.fit_transform(y)

    # Split — stratified to keep class balance
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )
    print(f"\n  Train: {len(X_train)} | Test: {len(X_test)}")

    # Preprocess
    X_train_p, y_train_p, X_test_p, scaler, pca = preprocess(
        X_train, y_train, X_test
    )

    # Train
    svm = train_svm(X_train_p, y_train_p)

    # Evaluate
    acc, prec, rec, f1 = evaluate_and_save(
        svm, scaler, pca, le, X_test_p, y_test
    )

    # Save
    save_models(svm, scaler, pca, le)

    total = time.time() - total_start
    mins, secs = divmod(int(total), 60)
    print(f"\nTotal time: {mins}m {secs}s")
    print(f"Final accuracy: {acc:.1%}")


# ─────────────────────────────────────────────
#  REAL-TIME WEBCAM DETECTION
# ─────────────────────────────────────────────
def run_detect():
    import cv2
    import numpy as np
    import joblib
    from skimage.feature import hog, local_binary_pattern

    model_path = os.path.join(MODEL_DIR, 'emotion_model.pkl')
    if not os.path.exists(model_path):
        print(f"\nERROR: Model not found at {model_path}")
        print("Run training first:  python emotion_svm_rafdb.py --train")
        return

    print(f"\nLoading model from {model_path} ...")
    bundle  = joblib.load(model_path)
    svm     = bundle['svm']
    scaler  = bundle['scaler']
    pca     = bundle['pca']
    le      = bundle['le']
    emotions = bundle.get('emotions', EMOTIONS)
    img_sz  = bundle.get('img_size', IMG_SIZE)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam (index 0).")
        print("Try: cv2.VideoCapture(1) if you have multiple cameras.")
        return

    print("Webcam open — press Q to quit\n")

    COLORS = {
        'happy':    (0,   220, 100),
        'sad':      (200, 100,  50),
        'angry':    (0,    60, 220),
        'surprise': (0,   200, 220),
        'fear':     (180,  60, 220),
        'disgust':  (60,  180,  60),
        'neutral':  (180, 180, 180),
    }

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
        )

        if len(faces) == 0:
            cv2.putText(frame, 'No face detected', (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)

        for (x, y, w, h) in faces:
            roi = cv2.resize(gray[y:y+h, x:x+w], img_sz)

            # ── Extract SAME features as training ──────────────
            hog_feat = hog(
                roi, orientations=9,
                pixels_per_cell=(4, 4), cells_per_block=(2, 2),
                block_norm='L2-Hys', visualize=False
            )
            lbp = local_binary_pattern(roi, P=8, R=1, method='uniform')
            lbp_hist, _ = np.histogram(
                lbp.ravel(), bins=np.arange(0, 11), density=True
            )
            feat = np.concatenate([hog_feat, lbp_hist]).reshape(1, -1)

            feat = scaler.transform(feat)
            feat = pca.transform(feat)

            label = le.inverse_transform(svm.predict(feat))[0]
            prob  = svm.predict_proba(feat).max()
            color = COLORS.get(label, (200, 200, 200))

            # Bounding box
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

            # Label pill background
            text = f'{label}  {prob:.0%}'
            (tw, th), _ = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2
            )
            cv2.rectangle(frame, (x, y-th-14), (x+tw+10, y), color, -1)
            cv2.putText(frame, text, (x+5, y-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

        cv2.putText(
            frame, 'SVM Emotion Detection (RAF-DB)  |  Q to quit',
            (10, frame.shape[0] - 12),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1
        )

        cv2.imshow('SVM Emotion Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Webcam closed.")


# ─────────────────────────────────────────────
#  EVALUATE ONLY (reload saved model)
# ─────────────────────────────────────────────
def run_evaluate():
    import joblib
    import numpy as np
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split

    model_path = os.path.join(MODEL_DIR, 'emotion_model.pkl')
    if not os.path.exists(model_path):
        print(f"ERROR: {model_path} not found. Run --train first.")
        return

    print(f"\nLoading model from {model_path} ...")
    bundle = joblib.load(model_path)
    svm    = bundle['svm']
    scaler = bundle['scaler']
    pca    = bundle['pca']
    le     = bundle['le']

    X, y = load_dataset()
    y_enc = le.transform(y)
    _, X_test, _, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )
    X_test_s = scaler.transform(X_test)
    X_test_p = pca.transform(X_test_s)

    evaluate_and_save(svm, scaler, pca, le, X_test_p, y_test)
    print("\nEvaluation complete. Charts saved to ./models/")


# ─────────────────────────────────────────────
#  MAIN ENTRY POINT
# ─────────────────────────────────────────────
def print_help():
    print("""
Usage:
  python emotion_svm_rafdb.py --install    Install all packages (run once)
  python emotion_svm_rafdb.py --train      Train SVM on RAF-DB
  python emotion_svm_rafdb.py --detect     Real-time webcam detection
  python emotion_svm_rafdb.py --evaluate   Re-evaluate saved model

Expected folder layout:
  Ai-Assignment/
  ├── emotion_svm_rafdb.py
  ├── RAF-DB/train/angry/ disgust/ fear/ happy/ neutral/ sad/ surprise/
  └── models/   ← auto-created
""")


if __name__ == '__main__':
    if len(sys.argv) < 2 or '--help' in sys.argv:
        print_help()

    elif '--install' in sys.argv:
        install_dependencies()

    elif '--train' in sys.argv:
        train_pipeline()

    elif '--detect' in sys.argv:
        run_detect()

    elif '--evaluate' in sys.argv:
        run_evaluate()

    else:
        print(f"Unknown argument: {sys.argv[1]}")
        print_help()