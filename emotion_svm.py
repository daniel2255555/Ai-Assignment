import os
import cv2
from skimage.feature import hog
import numpy as np
import joblib


model_path = r'C:\Users\User\Desktop\AiAssignment\Ai-Assignment\emotion_model_final.pkl'

# 1. Load the data
loaded_data = joblib.load(model_path)

# 2. Check if it's a dictionary and find the model
if isinstance(loaded_data, dict):
    print("Metadata found! Keys in your .pkl file:", loaded_data.keys())
    
    # Try common keys like 'model', 'svm', or 'classifier'
    # Change 'model' below to whatever key appeared in the print statement above
    if 'model' in loaded_data:
        model = loaded_data['model']
    elif 'svm' in loaded_data:
        model = loaded_data['svm']
    else:
        # If the key is something else, use the first key that isn't metadata
        first_key = list(loaded_data.keys())[0]
        model = loaded_data[first_key]
        print(f"Using key: {first_key}")
else:
    model = loaded_data

print("Model successfully extracted and ready for prediction.")

# 2. Load Face Detector (Haar Cascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Define emotion labels (Ensure these match your training set order)
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

def extract_hog_features(img):
    """
    Extract HOG features from a single grayscale image.
    Adjust pixels_per_cell and cells_per_block to match your training settings.
    """
    img_resized = cv2.resize(img, (64, 64)) # Match training image size
    features = hog(img_resized, 
                   orientations=9, 
                   pixels_per_cell=(8, 8), 
                   cells_per_block=(2, 2), 
                   visualize=False)
    return features.reshape(1, -1)

# 3. Start Video Capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # Extract the region of interest (face)
        roi_gray = gray[y:y+h, x:x+w]
        
        # Preprocess and Predict
        features = extract_hog_features(roi_gray)
        prediction = model.predict(features)[0]
        
        # If your model returns indices, map to label; if it returns strings, use directly
        label = emotion_labels[prediction] if isinstance(prediction, (int, np.integer)) else prediction

        # Draw results
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, str(label), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    cv2.imshow('Emotion Detection (SVM)', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()