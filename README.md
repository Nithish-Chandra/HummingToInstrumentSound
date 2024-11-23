# Humming Pitch Analysis and Modification

This project allows users to convert humming sounds into instrument-like sounds using machine learning techniques and sound processing algorithms.

## Features
Extracts pitch frequencies from audio using Librosa's piptrack function.
Groups pitch values into sets of 12.
Matches each set with the nearest target pitch value from a predefined database.
Reassigns pitch values to the nearest predefined values.
Modifies the audio based on these reassigned pitch values using a tunes database.
Saves modified audio files for each pitch set.

### Prerequisites
Ensure you have the following dependencies installed:

- Python 3.x
- Pip
- Required Python packages (listed below)

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/Nithish-Chandra/HummingToInstrumentSound.git
   cd HummingToInstrumentSound
   ```bash
   pip install librosa soundfile numpy

### Usage
1. Place the input audio file (Humming test.opus) in the directory
 /content/drive/MyDrive/Humming_Project/ or your own directory
2. Run the script:
   python3 Hum2Music.py
3. Modified audio files will be saved in your given path

### Output

Modified audio files named modified_audio_set_<set_number>.wav.
Each file corresponds to a processed set of pitch frequencies reassigned to the closest predefined value.

## Project Structure

```bash
humming-pitch-modification/
├── pitch_modification.py  # Main script
├── README.md              # Project documentation
├── data/
│   ├── Humming test.opus  # Input audio
│   ├── Tunes - 12 octaves/  # Tunes database
│       ├── A.mp3
│       ├── B.mp3
│       ├── ...
├── Modified_Audio/        # Output directory for modified audio files
