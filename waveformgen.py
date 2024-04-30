import os
import librosa
import librosa.display
import matplotlib.pyplot as plt

# Define the source and target directories
source_dir = '/workspace/voiceai/audio/uncommonvoice'
target_dir = '/workspace/voiceai/genimage'

# Check if the target directory exists, if not, create it
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

 
    
# Function to generate waveform images
def generate_waveform(audio_path, image_path):
    """
    Generate a waveform image from an audio file and save it to a specified path.
    """
    y, sr = librosa.load(audio_path)
    plt.figure(figsize=(14, 4), facecolor='white')  # Set background to white
    #librosa.display.waveshow(y, sr=sr)
    plt.plot(y, color='black')  # Set waveform color to black
    plt.axis('off')  # Remove axes
    plt.savefig(image_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(audio_path + " image path : " + image_path)

# Generate waveform images for all files in the source directory
for filename in os.listdir(source_dir):
    if filename.endswith('.wav'):  # Check if the file is a WAV audio file
        audio_path = os.path.join(source_dir, filename)
        image_filename = f"{os.path.splitext(filename)[0]}.png"  # Change file extension to .png
        image_path = os.path.join(target_dir, image_filename)
        generate_waveform(audio_path, image_path)
