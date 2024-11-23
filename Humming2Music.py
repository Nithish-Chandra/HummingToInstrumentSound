import librosa
import soundfile as sf
import numpy as np
import os

# Input audio path
audio_path = r"/content/drive/MyDrive/Humming_Project/Humming test.opus"

# Load the audio
y, sr = librosa.load(audio_path)

# Extract pitch frequencies
pitches, magnitudes = librosa.core.piptrack(y=y, sr=sr)
pitch_frequencies = pitches[pitches > 0]

# Group pitch frequencies into sets of 12
sets_of_pitch_values = [pitch_frequencies[i:i+12] for i in range(0, len(pitch_frequencies), 12)]

# Predefined pitch values
target_pitch_values = [440, 466.16, 493.88, 523.25, 554.37, 587.33, 
                       622.25, 659.26, 698.46, 739.99, 783.99, 830.61]

# Tunes database mapping
tunes_database = {
    440: "/content/drive/MyDrive/Humming_Project/Tunes - 12 octaves/A.mp3",
    466.16: "/content/drive/MyDrive/Humming_Project/Tunes - 12 octaves/A sharp #.mp3",
    493.88: "/content/drive/MyDrive/Humming_Project/Tunes - 12 octaves/B.mp3",
    523.25: "/content/drive/MyDrive/Humming_Project/Tunes - 12 octaves/C.mp3",
    554.37: "/content/drive/MyDrive/Humming_Project/Tunes - 12 octaves/C sharp or D - FLat.mp3",
    587.33: "/content/drive/MyDrive/Humming_Project/Tunes - 12 octaves/D.mp3",
    622.25: "/content/drive/MyDrive/Humming_Project/Tunes - 12 octaves/D sharp #.mp3",
    659.26: "/content/drive/MyDrive/Humming_Project/Tunes - 12 octaves/E.mp3",
    698.46: "/content/drive/MyDrive/Humming_Project/Tunes - 12 octaves/F.mp3",
    733.99: "/content/drive/MyDrive/Humming_Project/Tunes - 12 octaves/F Sharp #.mp3",
    783.99: "/content/drive/MyDrive/Humming_Project/Tunes - 12 octaves/G.mp3",
    830.61: "/content/drive/MyDrive/Humming_Project/Tunes - 12 octaves/G Sharp #.mp3"
}

# Output directory
output_directory = "/content/drive/MyDrive/Humming_Project/Modified_Audio/"
os.makedirs(output_directory, exist_ok=True)

# Process pitch sets
for index, pitch_set in enumerate(sets_of_pitch_values):
    mean_value = np.mean(pitch_set)
    nearest_value = min(target_pitch_values, key=lambda x: abs(x - mean_value))
    
    tune_path = tunes_database.get(nearest_value)
    if tune_path:
        tune_audio, _ = librosa.load(tune_path, sr=sr)
        modified_audio = tune_audio  # Placeholder for any additional modifications
        
        output_filename = os.path.join(output_directory, f"modified_audio_set_{index + 1}.wav")
        sf.write(output_filename, modified_audio, sr)
        print(f"Modified audio for pitch set {index + 1} saved to: {output_filename}")
    else:
        print(f"No tune found for pitch set {index + 1} with mean pitch {mean_value}")

