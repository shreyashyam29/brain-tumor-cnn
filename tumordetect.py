import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
# Define the path to the dataset
DATASET_PATH = "/kaggle/input/brain-mri-images-for-brain-tumor-detection"

# Set image parameters
IMG_SIZE = 128  # Resize images to 128x128 pixels
BATCH_SIZE = 32

# Function to load and preprocess images
def load_images_from_folder(folder):
    images = []
    labels = []
    for label, category in enumerate(["no", "yes"]):  # 'no' for no tumor, 'yes' for tumor
        path = os.path.join(folder, category)
        for filename in os.listdir(path):
            img = cv2.imread(os.path.join(path, filename))
            if img is not None:
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                img = img / 255.0  # Normalize pixel values
                images.append(img)
                labels.append(label)
    return np.array(images), np.array(labels)
# Load the dataset
X, y = load_images_from_folder(DATASET_PATH)

# Dataset Description
print(f"Total images: {len(X)}")
print(f"Number of 'no tumor' images: {np.sum(y == 0)}")
print(f"Number of 'tumor' images: {np.sum(y == 1)}")

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the Convolutional Neural Network (CNN) model
model = keras.Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),  # Convolutional layer with 32 filters, 3x3 kernel, and ReLU activation
    MaxPooling2D(2,2),  # Max pooling layer with 2x2 pool size
    
    Conv2D(64, (3,3), activation='relu'),  # Convolutional layer with 64 filters, 3x3 kernel, and ReLU activation
    MaxPooling2D(2,2),  # Max pooling layer with 2x2 pool size
    
    Conv2D(128, (3,3), activation='relu'),  # Convolutional layer with 128 filters, 3x3 kernel, and ReLU activation
    MaxPooling2D(2,2),  # Max pooling layer with 2x2 pool size

    Flatten(),  # Flatten layer to convert 2D matrix to 1D vector
    Dense(128, activation='relu'),  # Fully connected layer with 128 units and ReLU activation
    Dropout(0.5),  # Dropout layer to prevent overfitting
    Dense(1, activation='sigmoid')  # Output layer with sigmoid activation for binary classification
])
# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=BATCH_SIZE, validation_data=(X_test, y_test))

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc * 100:.2f}%")

# Predict on test set
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5).astype(int)  # Convert probabilities to binary predictions
# Calculate additional metrics
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
print("Confusion Matrix:")
print(conf_matrix)
# ROC-AUC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

# Plot Training and Validation Accuracy
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()

# Error Analysis
misclassified = np.where(y_pred.flatten() != y_test)[0]
plt.figure(figsize=(10, 5))
for i, idx in enumerate(misclassified[:5]):  # Display first 5 misclassified images
    plt.subplot(1, 5, i+1)
    plt.imshow(X_test[idx])
    plt.title(f"True: {y_test[idx]}, Pred: {y_pred[idx]}")
    plt.axis('off')
plt.show()

# Saving the trained model
model.save("/kaggle/working/brain_tumor_model.h5")