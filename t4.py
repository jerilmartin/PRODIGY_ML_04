import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt

# Step 1: Data Preprocessing

# Initialize variables
image_arrays = []
labels = []
main_directory_path = r'C:\Users\jeril\OneDrive\Desktop\prodigyml\task4\leapGestRecog\leapGestRecog'

# Load and preprocess images
for folder_name in os.listdir(main_directory_path):
    main_folder_path = os.path.join(main_directory_path, folder_name)
    if os.path.isdir(main_folder_path):
        for subfolder_name in os.listdir(main_folder_path):
            subfolder_path = os.path.join(main_folder_path, subfolder_name)
            if os.path.isdir(subfolder_path):
                for filename in os.listdir(subfolder_path):
                    image_path = os.path.join(subfolder_path, filename)
                    if filename.endswith('.png'):
                        try:
                            img = Image.open(image_path).convert('L')
                            img = img.resize((64, 64))  # Resize images to 64x64 pixels
                            img_array = np.array(img) / 255.0  # Normalize pixel values
                            image_arrays.append(img_array)
                            labels.append(folder_name)  # Use folder name as label
                        except Exception as e:
                            print(f"Error processing image: {filename}, Exception: {e}")

# Convert to numpy arrays
X = np.array(image_arrays)
y = np.array(labels)

# Encode labels
label_dict = {label: idx for idx, label in enumerate(np.unique(y))}
y_encoded = np.array([label_dict[label] for label in y])

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Reshape data for CNN
X_train = X_train.reshape(X_train.shape[0], 64, 64, 1)
X_test = X_test.reshape(X_test.shape[0], 64, 64, 1)

# Convert labels to categorical
y_train = to_categorical(y_train, num_classes=len(label_dict))
y_test = to_categorical(y_test, num_classes=len(label_dict))

# Step 2: Model Development

# Define the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(label_dict), activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Display the model architecture
model.summary()

# Step 3: Training and Evaluation

# Train the model
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2)

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_acc}')

# Step 4: Visualization

# Plot training & validation accuracy values
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.tight_layout()
plt.show()
