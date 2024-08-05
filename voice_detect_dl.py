import warnings
import os
import numpy as np
import tensorflow as tf
import librosa
import tensorflow_hub as hub
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
import joblib

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module='librosa')
warnings.filterwarnings("ignore", category=FutureWarning, module='librosa')
warnings.filterwarnings("ignore", message="PySoundFile failed. Trying audioread instead.")

# Load YAMNet model from local directory
yamnet_model = tf.saved_model.load('/Users/arundev/Downloads/yamnet-model')

# Function to augment audio
def augment_audio(y, sr):
    noise = np.random.randn(len(y))
    y_noisy = y + 0.005 * noise
    y_pitch_shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=2)
    y_time_stretched = librosa.effects.time_stretch(y, rate=1.2)
    return [y, y_noisy, y_pitch_shifted, y_time_stretched]

# Function to check if a file is an audio file
def is_audio_file(file_path):
    try:
        librosa.load(file_path, sr=None)
        return True
    except Exception:
        return False

# Function to extract embeddings from the dataset
def extract_embeddings(data_dir, yamnet_model):
    embeddings_list = []
    labels_list = []
    speakers = os.listdir(data_dir)
    for speaker in speakers:
        speaker_dir = os.path.join(data_dir, speaker)
        if not os.path.isdir(speaker_dir):
            continue
        for audio_file in os.listdir(speaker_dir):
            file_path = os.path.join(speaker_dir, audio_file)
            if not os.path.isfile(file_path) or not is_audio_file(file_path):
                continue
            y, sr = librosa.load(file_path, sr=None)
            augmentations = augment_audio(y, sr)
            for aug_y in augmentations:
                aug_y = librosa.resample(aug_y, orig_sr=sr, target_sr=16000)
                waveform = tf.convert_to_tensor(aug_y, dtype=tf.float32)
                _, embeddings, _ = yamnet_model(waveform)
                embeddings_list.append(embeddings.numpy().mean(axis=0))
                labels_list.append(speaker)
    return np.array(embeddings_list), np.array(labels_list)

# Load dataset and extract embeddings
data_dir = 'dataset'
x, y = extract_embeddings(data_dir, yamnet_model)

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(x, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Check for data leakage
train_set = set(map(tuple, X_train))
val_set = set(map(tuple, X_val))
test_set = set(map(tuple, X_test))
assert len(train_set.intersection(val_set)) == 0, "Train and Validation sets overlap!"
assert len(train_set.intersection(test_set)) == 0, "Train and Test sets overlap!"
assert len(val_set.intersection(test_set)) == 0, "Validation and Test sets overlap!"

# Define a more complex model to potentially improve performance
model = Sequential([
    Input(shape=(1024,)),
    Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    Dropout(0.5),
    Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    Dropout(0.5),
    Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    Dropout(0.5),
    Dense(len(np.unique(y_encoded)), activation='softmax')  # Dynamic number of output classes
])

# Compile the model
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_val, y_val))

# Save the model
model.save('speaker_detection_model.keras')

# Save the model and label encoder
joblib.dump(label_encoder, 'label_encoder.pkl')

"""
# Load the trained model and label encoder
model = tf.keras.models.load_model('speaker_detection_model.keras')
label_encoder = joblib.load('label_encoder.pkl')

# Make predictions on the validation set
val_predictions = model.predict(X_val)
val_predicted_labels = np.argmax(val_predictions, axis=1)
val_true_labels = y_val

# Print predictions and actual labels
for i in range(len(X_val)):
    print(f"Predicted: {label_encoder.inverse_transform([val_predicted_labels[i]])[0]}, Actual: {label_encoder.inverse_transform([val_true_labels[i]])[0]}")

# Print model summary and training history
model.summary()
print(f"Training history keys: {history.history.keys()}")

# Evaluate on test data
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_accuracy}")
"""
