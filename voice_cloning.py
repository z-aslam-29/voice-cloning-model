import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
import librosa

# Example values (adjust according to your audio data)
time_steps = 100  # Number of time steps in your audio data
num_features = 13  # Number of audio features (e.g., MFCCs) at each time step

# Example paths (replace with your actual paths)
project_directory = #path of your voice directory
train_subdirectory = "train"
train_directory = os.path.join(project_directory, train_subdirectory)

# Check if the train_directory exists
if not os.path.exists(train_directory):
    print(f"The directory '{train_directory}' does not exist.")

    # Ask the user if they want to create the directory
    create_directory = input("Do you want to create it? (yes/no): ").lower()

    if create_directory == 'yes':
        os.makedirs(train_directory)
        print(f"Directory '{train_directory}' created.")
    else:
        print("Please manually create the 'train' subdirectory in the 'Downloads' directory.")
        exit(1)

# Print debugging information
print("Project Directory:", project_directory)
print("Train Subdirectory:", train_subdirectory)
print("Train Directory:", train_directory)

# Load and preprocess audio data
def load_audio_data(directory, time_steps, num_features):
    audio_data = []
    for filename in os.listdir(directory):
        if filename.endswith(".wav"):
            file_path = os.path.join(directory, filename)
            try:
                print(f"Processing audio file: {file_path}")
                audio, _ = librosa.load(file_path, sr=None, mono=True)  # Load as mono
                if audio is None:
                    print(f"Error: Unable to load audio file '{file_path}' - audio is None")
                    continue

                # Extract features (MFCCs in this example)
                features = librosa.feature.mfcc(y=audio, sr=None, n_mfcc=num_features, n_fft=2048, hop_length=512)
                features = features[:, :time_steps]  # Trim or pad to the desired time_steps

                if features.size > 0:  # Check if the feature data is non-empty
                    print(f"Loaded audio file: {file_path}, Shape: {features.shape}")
                    audio_data.append(features)
                else:
                    print(f"Warning: Skipping empty feature data for '{file_path}'.")
            except Exception as e:
                print(f"Error loading audio file '{file_path}': {str(e)}")

    if not audio_data:
        print("Error: No non-empty feature data found. Please check your audio files.")
        exit(1)

    return np.array(audio_data)

# Print information before loading audio data
print("Loading audio data from:", train_directory)

# Adjusted loading function
X_train = load_audio_data(train_directory, time_steps, num_features)

# Check if the loaded audio data is empty
if X_train.size == 0:
    print("Error: No non-empty feature data found. Please check your audio files.")
    exit(1)

# Define the Voice Cloning Model
def create_voice_cloning_model(input_shape, latent_dim):
    model = models.Sequential()

    # Encoder
    model.add(layers.InputLayer(input_shape=input_shape))
    model.add(layers.Conv1D(32, kernel_size=5, strides=2, padding="same", activation="relu"))
    model.add(layers.Conv1D(64, kernel_size=5, strides=2, padding="same", activation="relu"))
    model.add(layers.Conv1D(128, kernel_size=5, strides=2, padding="same", activation="relu"))
    model.add(layers.Flatten())
    model.add(layers.Dense(latent_dim, activation="relu"))

    # Decoder
    model.add(layers.Dense(np.prod(input_shape), activation="relu"))
    model.add(layers.Reshape(input_shape))

    return model

# Check if the loaded audio data is empty
Y_train = X_train  # Assuming you want to reconstruct the input as output

# Create and compile the model
input_shape = (time_steps, num_features)
latent_dim = 256
voice_cloning_model = create_voice_cloning_model(input_shape, latent_dim)
voice_cloning_model.compile(optimizer=Adam(learning_rate=0.001), loss=MeanSquaredError())

# Train the model
voice_cloning_model.fit(X_train, Y_train, epochs=50, batch_size=32)

# Save the trained model
voice_cloning_model.save("voice_cloning_model.h5")
