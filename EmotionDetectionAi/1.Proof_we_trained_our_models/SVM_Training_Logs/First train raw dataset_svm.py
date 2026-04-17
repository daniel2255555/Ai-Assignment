import os
import sys
import cv2
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.feature import hog, local_binary_pattern
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE 

# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────
DATA_DIR       = rC:\Users\User\Desktop\AiAssignment\Ai-Assignment\RAF-DB\train' 
# Local path for the model file
DESKTOP_PATH   = os.path.join(os.path.expanduser("~"), "Desktop")
SAVE_PATH      = os.path.join(DESKTOP_PATH, 'emotion_model_final.pkl')

IMG_SIZE       = (64, 64) 
EMOTIONS       = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'] # Fixed 'shock' to 'surprise' to match RAF-DB

def get_features(img):
    h_feat = hog(img, orientations=9, pixels_per_cell=(4, 4), 
                cells_per_block=(2, 2), block_norm='L2-Hys')
    lbp = local_binary_pattern(img, P=8, R=1, method='uniform')
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=10, range=(0, 10), density=True)
    return np.concatenate([h_feat, lbp_hist])

def load_data(data_dir):
    print(f"\n[1/5] Loading images from: {data_dir}")
    X, y = [], []
    for emotion in EMOTIONS:
        folder = os.path.join(data_dir, emotion)
        if not os.path.exists(folder):
            print(f"!! Missing: {folder}")
            continue
        files = [f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg'))][:1500]
        for fname in files:
            img = cv2.imread(os.path.join(folder, fname), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, IMG_SIZE)
                X.append(get_features(img))
                y.append(emotion)
    return np.array(X), np.array(y)

def plot_accuracy_flow(model, X, y):
    print("\n[5/5] Generating Accuracy Flow...")
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=2, n_jobs=1, train_sizes=[0.3, 0.6, 1.0])
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', label="Training Accuracy")
    plt.plot(train_sizes, np.mean(test_scores, axis=1), 'o-', label="Validation Accuracy")
    plt.title("Accuracy Improvement Flow")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(DESKTOP_PATH, "accuracy_flow.png"))
    plt.show()

if __name__ == '__main__':
    X_raw, y_raw = load_data(DATA_DIR)
    if len(X_raw) == 0:
        print("Error: No images found. Check path.")
        sys.exit()

    le = LabelEncoder()
    y_enc = le.fit_transform(y_raw)
    X_train, X_test, y_train, y_test = train_test_split(X_raw, y_enc, test_size=0.2, stratify=y_enc, random_state=42)

    # 1. Scale
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    # 2. PCA
    pca = PCA(n_components=0.95, whiten=True)
    X_train_pca = pca.fit_transform(X_train_s)
    X_test_pca = pca.transform(X_test_s)
    
    # 3. SMOTE
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train_pca, y_train)

    # 4. Train
    print("\n[4/5] Training SVM...")
    model = SVC(kernel='rbf', C=10, probability=True, class_weight='balanced')
    model.fit(X_res, y_res)

    # 5. SAVE THE MODEL (Crucial Step)
    print(f"\n[SAVING] Exporting model bundle to {SAVE_PATH}...")
    model_bundle = {
        'model': model,
        'scaler': scaler,
        'pca': pca,
        'le': le
    }
    joblib.dump(model_bundle, SAVE_PATH)
    print("✅ Model bundle saved successfully!")

    # 6. Evaluate
    y_pred = model.predict(X_test_pca)
    print("\nFinal Metrics:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    plot_accuracy_flow(model, X_res, y_res)