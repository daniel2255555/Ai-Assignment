import tensorflow as tf
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D

physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        # Tell TF to dynamically grow GPU memory usage
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print("GPU Memory Growth Enabled!")
    except RuntimeError as e:
        print(e)

print(f"GPUs detected: {len(physical_devices)}")
print(f"GPUs detected: {len(tf.config.list_physical_devices('GPU'))}")

# ==========================================
# 1. LOAD IMAGES
# ==========================================
# Ensure this points to the ORIGINAL, clean dataset to get your 79% back!
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,       
    zoom_range=0.10,         
    width_shift_range=0.15,
    height_shift_range=0.15
)
test_datagen = ImageDataGenerator(rescale=1./255)

# Connect generators 
train_generator = train_datagen.flow_from_directory(
    'dataset/DATASET/train', 
    target_size=(224, 224), 
    color_mode="rgb", 
    batch_size=16, 
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    'dataset/DATASET/test', 
    target_size=(224, 224), 
    color_mode="rgb", 
    batch_size=16, 
    class_mode='categorical', 
    shuffle=False
)

# ==========================================
# 2. ARCHITECTURE (MobileNetV2)
# ==========================================
print("\nDownloading Google's MobileNetV2 brain...")

base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False 

model = Sequential([
    base_model,
    GlobalAveragePooling2D(), 
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(7, activation='softmax') 
])

lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
early_stopper = EarlyStopping(monitor='val_accuracy', patience=12, restore_best_weights=True)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# ==========================================
# 4. PHASE 1: WARM-UP TRAINING
# ==========================================
print("Starting Advanced CNN training on your local GPU (Phase 1)...")
history = model.fit(
    train_generator,
    epochs=30,
    validation_data=test_generator,
    callbacks=[lr_reducer, early_stopper],
    workers=4,          
    max_queue_size=20   
)

model.save('cnn_emotion_model_phase1.h5')
print("Phase 1 CNN Model saved successfully!")

# ==========================================
# 5. PHASE 2: FINE-TUNING (100% UNFREEZE)
# ==========================================
print("\nStarting Phase 2: Fine-Tuning the Google Brain...")

base_model.trainable = True

from tensorflow.keras.optimizers import Adam
model.compile(
    optimizer=Adam(learning_rate=0.00001), 
    loss='categorical_crossentropy', 
    metrics=['accuracy']
)

history_fine = model.fit(
    train_generator,
    epochs=40, 
    validation_data=test_generator,
    callbacks=[early_stopper],
    workers=4,          
    max_queue_size=20   
)

model.save('cnn_emotion_model_FINAL.h5')
print("Phase 2 Complete. Ultimate Model Saved!")


# ==========================================
# 6. EVALUATION & AUTO-SAVE SECTION
# ==========================================
print("\nGenerating and saving final evaluation graphs...")

acc = history.history['accuracy'] + history_fine.history['accuracy']
val_acc = history.history['val_accuracy'] + history_fine.history['val_accuracy']
loss = history.history['loss'] + history_fine.history['loss']
val_loss = history.history['val_loss'] + history_fine.history['val_loss']

# A. Plot and Auto-Save Training & Validation Curves
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(acc, label='Train Accuracy')
axes[0].plot(val_acc, label='Test Accuracy')
axes[0].axvline(x=len(history.history['accuracy'])-1, color='red', linestyle='--', label='Start Phase 2')
axes[0].set_title('CNN Model Accuracy (Phase 1 & 2)')
axes[0].set_ylabel('Accuracy')
axes[0].set_xlabel('Epoch')
axes[0].legend(loc='lower right')
axes[0].grid(True)

axes[1].plot(loss, label='Train Loss')
axes[1].plot(val_loss, label='Test Loss')
axes[1].axvline(x=len(history.history['loss'])-1, color='red', linestyle='--', label='Start Phase 2')
axes[1].set_title('CNN Model Loss (Phase 1 & 2)')
axes[1].set_ylabel('Loss')
axes[1].set_xlabel('Epoch')
axes[1].legend(loc='upper right')
axes[1].grid(True)

# ---> AUTO SAVE GRAPHS <---
fig.savefig('training_history_graphs.png', bbox_inches='tight')
print("✅ Saved: training_history_graphs.png")
plt.close(fig) # Closes the figure to free up memory


# B. Generate Predictions
print("\nGenerating predictions for classification report...")
Y_pred = model.predict(test_generator) 
y_pred = np.argmax(Y_pred, axis=1) 
y_true = test_generator.classes 
class_labels = list(test_generator.class_indices.keys())

# C. Print and Auto-Save Classification Report 
print("\n" + "="*50)
print("FINAL CLASSIFICATION REPORT")
print("="*50)
report_text = classification_report(y_true, y_pred, target_names=class_labels)
print(report_text)

# ---> AUTO SAVE REPORT <---
with open('final_classification_report.txt', 'w') as f:
    f.write("FINAL CLASSIFICATION REPORT\n")
    f.write("==================================================\n")
    f.write(report_text)
print("✅ Saved: final_classification_report.txt")

# D. Plot and Auto-Save the Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.title('Emotion Detection Confusion Matrix (FINAL)')
plt.ylabel('True Emotion (Actual)')
plt.xlabel('Predicted Emotion (AI Guess)')

# ---> AUTO SAVE MATRIX <---
plt.savefig('confusion_matrix.png', bbox_inches='tight')
print("✅ Saved: confusion_matrix.png")
plt.close()

print("\n🎉 Training Complete! All files have been saved to your project folder.")
