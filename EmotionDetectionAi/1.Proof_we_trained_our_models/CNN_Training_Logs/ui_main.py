import cv2
import numpy as np
import tkinter as tk
from tkinter import Label, Frame
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model

# 1. Load AI and Face Detector
print("Loading UI and MobileNetV2 AI Model...")
model = load_model('cnn_emotion_model_FINAL.h5')
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 2. Setup the Main UI Window
root = tk.Tk()
root.title("BingXue Emotion Detection Prototype")
root.geometry("800x700")
root.configure(bg="#2c3e50") # Dark modern background

# UI Header
header = Label(root, text="Real-Time Emotion Detector", font=("Helvetica", 24, "bold"), bg="#2c3e50", fg="white")
header.pack(pady=20)

# Frame to hold the video
video_frame = Frame(root, bg="black", bd=5)
video_frame.pack()

# Label that will actually show the video frames
video_label = Label(video_frame)
video_label.pack()

# Open Webcam
cap = cv2.VideoCapture(0)

def update_frame():
    ret, frame = cap.read()
    if ret:
        # Flip the frame so it acts like a mirror
        frame = cv2.flip(frame, 1)
        
        # We need grayscale ONLY for the face detector to find the box
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
            
            # ==========================================
            # UPGRADED PREDICTION LOGIC FOR MOBILENETV2
            # ==========================================
            # 1. Crop from the COLOR frame, not the gray frame!
            roi_color = frame[y:y+h, x:x+w]
            
            # 2. Convert BGR (OpenCV) to RGB (What MobileNetV2 expects)
            roi_rgb = cv2.cvtColor(roi_color, cv2.COLOR_BGR2RGB)
            
            # 3. Resize to high-definition 224x224
            roi_resized = cv2.resize(roi_rgb, (224, 224))
            
            # 4. Normalize the pixels (0 to 1)
            roi_normalized = roi_resized / 255.0
            
            # 5. Add the batch dimension -> shape becomes (1, 224, 224, 3)
            roi_ready = np.expand_dims(roi_normalized, axis=0)
            # ==========================================
            
            predictions = model.predict(roi_ready, verbose=0)
            max_index = np.argmax(predictions[0])
            predicted_emotion = emotion_labels[max_index]
            confidence = round(predictions[0][max_index] * 100)
            
            display_text = f"{predicted_emotion} {confidence}%"
            cv2.putText(frame, display_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

        # Convert the OpenCV frame (BGR) to a UI-friendly image (RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb_frame)
        imgtk = ImageTk.PhotoImage(image=img)
        
        # Put the image into the UI label
        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)
        
    # Schedule the next frame update in 10 milliseconds
    video_label.after(10, update_frame)

def on_close():
    """Handles safely shutting down the camera and UI."""
    cap.release()
    root.destroy()

# Add a quit button
quit_btn = tk.Button(root, text="Stop Camera & Exit", font=("Helvetica", 14), bg="#e74c3c", fg="white", command=on_close)
quit_btn.pack(pady=20)

# Ensure camera turns off if they click the window's 'X' button
root.protocol("WM_DELETE_WINDOW", on_close)

# Start the continuous video loop
update_frame()
root.mainloop()
