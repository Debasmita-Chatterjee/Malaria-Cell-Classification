# malaria_cnn_model.py
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

from Preprocessing import load_data

# ------------------------------
# STEP 2: MAIN EXECUTION
# ------------------------------
if __name__ == "__main__":
    # Define paths
    Infected_path = "cell_images/cell_images/Parasitized"
    Uninfected_path = "cell_images/cell_images/Uninfected"

    # Load the data
    images, labels = load_data(Infected_path, Uninfected_path)
    print("✅ Data Loaded Successfully!")
    print(f"✅ Loaded {len(images)} images with labels.")
    print(f"🧩 Image shape: {images[0].shape}")

    # Normalize pixel values
    images = images.astype('float32') / 255.0

    # One-hot encode labels
    labels = to_categorical(labels)

    # ------------------------------
    # STEP 3: SPLIT DATA (70/30)
    # ------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        images, labels, test_size=0.3, random_state=42, stratify=labels
    )
    print(f"📊 Training samples: {X_train.shape[0]}")
    print(f"📊 Testing samples: {X_test.shape[0]}")

    # ------------------------------
    # STEP 4: BUILD CNN MODEL
    # ------------------------------
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),

        Flatten(),

        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(2, activation='softmax')
    ])

    # ------------------------------
    # STEP 5: COMPILE MODEL
    # ------------------------------
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # ------------------------------
    # STEP 6: TRAIN MODEL
    # ------------------------------
    history = model.fit(X_train, y_train, validation_split=0.2, epochs=10, batch_size=32)

    # ------------------------------
    # STEP 7: DATA VISUALIZATION
    # ------------------------------
    plt.figure(figsize=(10, 4))

    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # ------------------------------
    # STEP 8: SAVE MODEL
    # ------------------------------
    model.save("malaria_cnn_model.h5")
    print("💾 Model saved as 'malaria_cnn_model.h5'")

