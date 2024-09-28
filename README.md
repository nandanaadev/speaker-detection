# Speaker Detection System

## Overview

This project implements a speaker detection system using deep learning techniques. The system is designed to identify different speakers based on audio samples. It extracts audio embeddings using the YAMNet model and then employs a custom neural network built with TensorFlow for speaker classification.

## Features

- Audio file selection through a simple GUI.
- Real-time prediction of speakers from uploaded audio files.
- Data augmentation techniques to enhance the robustness of the model.
- Ability to train the model on new datasets.

## Tech Stack

- **Programming Language**: Python
- **Libraries**:
  - **TensorFlow**: For building and training the deep learning model.
  - **Librosa**: For audio processing and feature extraction.
  - **TensorFlow Hub**: For loading the YAMNet model for audio embeddings.
  - **Scikit-learn**: For data preprocessing and encoding labels.
  - **Joblib**: For saving and loading models and encoders.
  - **Tkinter**: For building a simple user interface to select audio files for prediction.
