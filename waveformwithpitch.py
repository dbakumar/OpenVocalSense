import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# Define the source and target directories
source_dir = '/workspace/voiceai/audio/uncommonvoice'
target_dir = '/workspace/voiceai/genimagewithpitchgray'

# Check if the target directory exists, if not, create it
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

# Function to generate waveform and pitch images
def generate_waveform_and_pitch(audio_path, image_path):
    """
    Generate a waveform image and a pitch (spectrogram) image from an audio file and save it to a specified path.
    """
    y, sr = librosa.load(audio_path)
    
    # Create a figure with two subplots
    fig, axs = plt.subplots(2, 1, figsize=(14, 8), facecolor='white')
    
    # Plot the amplitude waveform
    axs[0].plot(y, color='black')
    axs[0].axis('off')  # Remove axes
    
    # Plot the spectrogram for pitch
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    img = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log', ax=axs[1], cmap='gray_r')
    axs[1].axis('off')  # Remove axes
    
    # Save the figure
    plt.savefig(image_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"{audio_path} image path: {image_path}")

# Generate waveform and pitch images for all files in the source directory
for filename in os.listdir(source_dir):
    if filename.endswith('.wav'):  # Check if the file is a WAV audio file
        audio_path = os.path.join(source_dir, filename)
        image_filename = f"{os.path.splitext(filename)[0]}.png"  # Change file extension to .png
        image_path = os.path.join(target_dir, image_filename)
        generate_waveform_and_pitch(audio_path, image_path)
