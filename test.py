import warnings
import librosa
import tensorflow as tf
import numpy as np
import joblib
import tensorflow_hub as hub
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module='librosa')
warnings.filterwarnings("ignore", category=FutureWarning, module='librosa')
warnings.filterwarnings("ignore", message="PySoundFile failed. Trying audioread instead.")

# Load YAMNet model from local directory
yamnet_model = tf.saved_model.load('/Users/arundev/Downloads/yamnet-model')

# Function to resample and preprocess audio
def preprocess_audio(audio, sample_rate):
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)
    audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
    return audio

# Function to predict speaker from an audio file
def predict_speaker(audio, model, label_encoder, sample_rate=16000):
    waveform = preprocess_audio(audio, sample_rate)
    waveform = tf.convert_to_tensor(waveform, dtype=tf.float32)
    
    try:
        # Get embeddings using YAMNet
        _, embeddings, _ = yamnet_model(waveform)
        embedding = embeddings.numpy().mean(axis=0).reshape(1, -1)  # Reshape to match input shape
        
    except Exception as e:
        print(f"Error while extracting embeddings: {e}")
        return None
    
    # Make prediction using the loaded model
    predictions = model.predict(embedding)

    # Print prediction info
    print("Predictions:", predictions)
    
    predictions = np.squeeze(predictions)  # Remove single-dimensional entries
    predicted_label = np.argmax(predictions, axis=0)
    
    # Print predicted label
    print("Predicted label:", predicted_label)
    
    # Map the predicted label to the speaker name
    speaker_name = label_encoder.inverse_transform([predicted_label])
    
    return speaker_name[0]

# Load the trained model
model = tf.keras.models.load_model('speaker_detection_model.keras')

# Load the label encoder
label_encoder = joblib.load('label_encoder.pkl')

# Function to select an audio file
def select_audio_file():
    Tk().withdraw()  # Hide the main window
    filename = askopenfilename(filetypes=[("Audio Files", "*.wav *.mp3 *.m4a")])
    return filename

# Select an audio file
for i in range(14):
    audio_file_path = select_audio_file()
    if audio_file_path:
        print(f"Loaded audio file: {audio_file_path}")
        audio, sample_rate = librosa.load(audio_file_path, sr=None)
        print(f"Sample rate: {sample_rate}")
        
        # Make prediction
        speaker_name = predict_speaker(audio, model, label_encoder, sample_rate)
        print(f"Predicted Speaker: {speaker_name}")
    else:
        print("No file selected.")
