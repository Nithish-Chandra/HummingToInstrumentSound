# HummingToInstrumentSound
Our project aims to create a user-friendly system that can turn human voice or humming  sounds into different musical instrument sounds. This means that if you sing or hum a tune  into our system, it will automatically identify the main musical note you are singing and then the corresponding sound of a specific instrument, like a guitar or piano.
!pip install --upgrade tensorflow 
 Install all the required libraries.
!pip install matplotlib 
!pip install librosa 
!pip install playsound 
!pip install pyaudio 
!pip install soundfile 
!pip install aubio 
!pip install pygame 
7 
 Import all the downloaded libraries.
import librosa 
import numpy as np 
import matplotlib.pyplot as plt 
import soundfile as sf 
import os 
3.2.2. Pitch Detection: 
 Calculating highest and lowest frequencies in an audio 
import librosa 
import numpy as np 
# Load the audio file
audio_path = r"/content/drive/MyDrive/Humming_Project/Humming test.opus"
y, sr = librosa.load(audio_path) 
# Extract pitch information using Librosa's pitch detection function
pitches, magnitudes = librosa.core.piptrack(y=y, sr=sr) 
# Get the pitch frequencies (non-zero values from the pitches array)
pitch_frequencies = pitches[pitches > 0] 
# Find the highest and lowest pitch frequencies
highest_pitch = np.max(pitch_frequencies) 
lowest_pitch = np.min(pitch_frequencies) 
print("Highest Pitch Frequency (Hz):", highest_pitch) 
print("Lowest Pitch Frequency (Hz):", lowest_pitch) 
Output: 
Highest Pitch Frequency (Hz): 3995.184 
Lowest Pitch Frequency (Hz): 145.58928
 Calculating mean pitch and plotting time domain graph 
import matplotlib.pyplot as plt 
import librosa 
# Load the audio file
audio_path = r"/content/drive/MyDrive/Humming_Project/Humming test.opus"
y, sr = librosa.load(audio_path) 
# Extract pitch information using Librosa's pitch detection function
pitches, magnitudes = librosa.piptrack(y=y, sr=sr) 
# Calculate the mean pitch
8 
mean_pitch = pitches.mean() 
# Plot the time domain graph
plt.figure(figsize=(10, 4)) 
plt.plot(y) 
plt.title("Time Domain Graph") 
plt.xlabel("Time (samples)") 
plt.ylabel("Amplitude") 
plt.show() 
print("Mean pitch:", mean_pitch) 
Output: 
Mean pitch: 5.4070015
 Assigning Pitch values to sets of audio file 
import librosa 
# Load the audio file
audio_path = r"/content/drive/MyDrive/Humming_Project/Humming test.opus"
y, sr = librosa.load(audio_path) 
# Extract pitch information using Librosa's pitch detection function
pitches, magnitudes = librosa.core.piptrack(y=y, sr=sr) 
# Get the pitch frequencies (non-zero values from the pitches array)
pitch_frequencies = pitches[pitches > 0] 
# Separate pitch values into sets of 12 consecutive values
sets_of_pitch_values = [pitch_frequencies[i:i+12] for i in range(0, 
len(pitch_frequencies), 12)] 
# Print the sets of pitch values
9 
print("Sets of Pitch Frequencies:") 
for index, pitch_set in enumerate(sets_of_pitch_values): 
 print(f"Set {index + 1}: {pitch_set}") 
Output: 
Sets of Pitch Frequencies: 
Set 1: [146.95204 147.36438 149.66182 152.7133 155.52838 153.27415 
153.84789 
 154.89957 156.11026 155.8066 155.38635 155.5817 ] 
Set 2: [156.05772 154.06062 149.6059 145.63385 150.94595 151.77693 
149.4784 
 149.20107 149.07593 149.13168 149.9259 150.45569] 
Set 3: [151.13818 151.10751 151.5519 152.04303 152.58257 151.91762 
151.54747 
 151.93617 152.64844 150.86243 149.5184 145.58928] 
Set 4: [148.06818 161.38824 160.9452 157.70805 164.4982 159.92513 
160.59674 
 156.23485 156.87218 156.88924 156.39357 156.83951] 
Set 5: [156.57274 156.77528 156.7562 156.39622 156.9182 156.5835 
158.43355 
 158.28787 163.55618 162.39928 163.79742 166.57707] 
Set 6: [173.7514 169.73674 174.19987 173.34059 175.1518 169.77585 
177.23929 
 169.47025 176.28014 174.24107 176.01367 176.86613] 
Set 7: [175.7975 168.97835 181.25276 181.85059 180.1362 180.55173 
181.5587 
 182.6513 183.45712 183.59637 183.82248 182.68657] 
Set 8: [179.41693 179.071 182.18088 185.97908 183.39224 180.43365 
181.5513 
 188.09525 186.62538 186.12187 184.34367 183.45001] 
Set 9: [186.35349 181.89742 196.1709 192.30385 194.55452 193.18338 
198.11865 
 188.99786 196.95145 195.19803 194.5271 195.80861] 
Set 10: [191.22667 198.9653 192.68112 188.92113 199.95647 206.15872 
204.81331 
 208.24591 207.15126 204.55508 207.65114 208.92686] 
Set 11: [209.29016 206.97432 204.51443 204.99088 208.08347 207.2052 
205.22762 
 204.96683 201.00563 204.45529 204.88943 200.41689] 
Set 12: [200.42558 203.55838 201.79642 199.6636 211.33318 218.5932 
215.05026 
 213.51306 211.31422 213.19838 210.02402 218.57137] 
Set 13: [225.27394 229.96559 223.83391 228.11508 227.0553 222.2738 
224.95526 
 221.18802 222.85426 222.13051 223.0845 223.45602] 
Set 14: [222.03325 221.07422 229.5164 234.87047 234.70662 233.15683 
236.29062 
 234.80711 233.89041 235.80106 232.51743 232.5048 ] 
Set 15: [241.42465 239.33395 235.68991 235.68767 232.61574 240.07074 
233.55226 
10 
 244.43889 249.41127 245.38185 245.07143 247.51706] 
Set 16: [248.40672 250.63698 248.25873 248.36604 250.25279 252.22813 
248.08612 
 247.2054 249.54471 250.82892 249.9369 252.27454] 
Set 17: [250.42708 252.95755 252.95897 251.99371 249.9322 260.99167 
257.54987 
 258.50153 258.1079 258.0773 255.89517 257.01547] 
Set 18: [258.08945 259.43945 255.56346 262.04355 254.64804 253.1943 
253.87367 
 255.00537 253.09502 254.00914 253.22223 253.45529] 
Set 19: [270.2453 269.39868 270.5064 265.99576 271.35278 270.9872 
268.3318 
 269.30475 266.56845 267.133 270.3266 283.1462 ] 
. 
. 
. 
. 
Set 54: [1394.5844 1409.8528 1419.1967 1445.3241 1445.1482 1473.9655 
1505.2941 
 1507.5188 1551.3436 1552.608 1583.3043 1601.7177] 
Set 55: [1613.4895 1646.4397 1652.2485 1691.3882 1721.2563 1747.1193 
1766.9718 
 1799.1273 1803.9541 1815.4182 1846.001 1903.457 ] 
Set 56: [1903.9525 1921.9572 1929.795 1957.8514 1965.6416 2037.5161 
2037.379 
 2071.2683 2079.8367 2094.3523 2127.983 2139.596 ] 
Set 57: [2188.2244 2222.4355 2215.4878 2240.9163 2268.9604 2293.3499 
2303.2217 
 2334.8584 2366.115 2364.7722 2402.699 2426.7112] 
Set 58: [2423.3916 2463.8076 2463.0286 2501.1865 2502.7957 2516.3132 
2561.6218 
 2561.4248 2598.8674 2617.601 2657.2302 2672.6672] 
Set 59: [2720.4543 2721.355 2758.695 2752.0232 2797.6929 2798.8167 
2836.1458 
 2830.2664 2877.2195 2877.4663 2908.7986 2920.8115] 
Set 60: [2938.198 2980.517 2991.06 3027.7488 3030.7737 3080.8662 
3089.1194 
 3122.7913 3121.922 3151.2803 3184.5903 3189.964 ] 
Set 61: [3211.577 3224.602 3261.576 3266.5132 3298.1594 3296.412 
3343.6272 
 3343.1084 3381.853 3379.9326 3411.2725 3414.4224] 
Set 62: [3445.772 3442.9873 3481.2817 3491.1104 3516.0679 3528.399 
3537.143 
 3576.7915 3575.4207 3607.414 3632.8232 3635.3958] 
Set 63: [3665.9822 3662.914 3695.3042 3696.5251 3734.1616 3735.761 
3768.3723 
 3796.4019 3796.2805 3827.1587 3829.9504 3861.2322] 
Set 64: [3872.3281 3909.7473 3904.608 3933.2422 3942.3274 3957.4712 
3969.6816 
 3995.184 ] 
11 
 Displaying modified sets of pitch frequencies 
import librosa 
# Load the audio file
audio_path = r"/content/drive/MyDrive/Humming_Project/Humming test.opus"
y, sr = librosa.load(audio_path) 
# Extract pitch information using Librosa's pitch detection function
pitches, magnitudes = librosa.core.piptrack(y=y, sr=sr) 
# Get the pitch frequencies (non-zero values from the pitches array)
pitch_frequencies = pitches[pitches > 0] 
# Separate pitch values into sets of 12 consecutive values
sets_of_pitch_values = [pitch_frequencies[i:i+12] for i in range(0, 
len(pitch_frequencies), 12)] 
# Define the target pitch values for reassigning
target_pitch_values = [440, 466.16, 493.88, 523.25, 554.37, 587.33, 
622.25, 659.26, 698.46, 739.99, 783.99, 830.61] 
# Iterate through each set and reassign pitch values
for pitch_set in sets_of_pitch_values: 
 min_pitch = min(pitch_set) 
 max_pitch = max(pitch_set) 
 # Reassign pitch values based on their position within the set
 for i in range(len(pitch_set)): 
 normalized_pitch = (pitch_set[i] - min_pitch) / (max_pitch - 
min_pitch) 
 index = int(normalized_pitch * (len(target_pitch_values) - 1)) 
 pitch_set[i] = target_pitch_values[index] 
# Print the modified sets of pitch values
print("Modified Sets of Pitch Frequencies:") 
for index, pitch_set in enumerate(sets_of_pitch_values): 
 print(f"Set {index + 1}: {pitch_set}") 
Output: 
Modified Sets of Pitch Frequencies: 
Set 1: [440. 440. 523.25 622.25 783.99 659.26 698.46 739.99 830.61 
783.99 
 783.99 783.99] 
Set 2: [830.61 698.46 554.37 440. 587.33 622.25 554.37 523.25 523.25 
523.25 
 554.37 587.33] 
Set 3: [698.46 698.46 739.99 783.99 783.99 739.99 739.99 739.99 830.61 
698.46 
 622.25 440. ] 
Set 4: [440. 698.46 698.46 622.25 830.61 659.26 698.46 587.33 587.33 
587.33 
12 
 587.33 587.33] 
Set 5: [440. 440. 440. 440. 440. 440. 493.88 493.88 659.26 
622.25 
 659.26 830.61] 
Set 6: [622.25 440. 622.25 587.33 698.46 440. 830.61 440. 739.99 
622.25 
 739.99 783.99] 
Set 7: [587.33 440. 739.99 739.99 698.46 698.46 739.99 783.99 783.99 
783.99 
 830.61 783.99] 
Set 8: [440. 440. 523.25 698.46 587.33 466.16 523.25 830.61 739.99 
698.46 
 622.25 587.33] 
Set 9: [523.25 440. 739.99 659.26 698.46 659.26 830.61 554.37 783.99 
739.99 
 698.46 739.99] 
Set 10: [466.16 587.33 493.88 440. 622.25 739.99 698.46 783.99 783.99 
698.46 
 783.99 830.61] 
. 
. 
. 
. 
Set 61: [440. 440. 493.88 493.88 554.37 554.37 659.26 659.26 739.99 
739.99 
 783.99 830.61] 
Set 62: [440. 440. 493.88 493.88 554.37 554.37 587.33 659.26 659.26 
739.99 
 783.99 830.61] 
Set 63: [440. 440. 466.16 466.16 523.25 554.37 587.33 659.26 659.26 
739.99 
 739.99 830.61] 
Set 64: [440. 523.25 493.88 587.33 622.25 659.26 698.46 830.61] 
3.2.3.Dividing the audio sets into 12octaves. 
import librosa 
import numpy as np 
# Load the audio file
audio_path = r"/content/drive/MyDrive/Humming_Project/Humming test.opus"
y, sr = librosa.load(audio_path) 
# Extract pitch information using Librosa's pitch detection function
pitches, magnitudes = librosa.core.piptrack(y=y, sr=sr) 
# Get the pitch frequencies (non-zero values from the pitches array)
pitch_frequencies = pitches[pitches > 0] 
# Separate pitch values into sets of 12 consecutive values
13 
sets_of_pitch_values = [pitch_frequencies[i:i+12] for i in range(0, 
len(pitch_frequencies), 12)] 
# Define the target pitch values for reassigning
target_pitch_values = [440, 466.16, 493.88, 523.25, 554.37, 587.33, 
622.25, 659.26, 698.46, 739.99, 783.99, 830.61] 
# Iterate through each set, find the mean value, and assign the entire 
set with the mean
for index, pitch_set in enumerate(sets_of_pitch_values): 
mean_value = np.mean(pitch_set) 
# Assign the entire set with the mean value
pitch_set[:] = [mean_value] * len(pitch_set) 
# Find the nearest value from the target_pitch_values list
nearest_value = min(target_pitch_values, key=lambda x: abs(x - 
mean_value)) 
# Assign the entire set with the nearest value from the 
target_pitch_values list
sets_of_pitch_values[index][:] = [nearest_value] * len(pitch_set) 
# Print the modified sets of pitch values
print("Modified Sets of Pitch Frequencies:") 
for index, pitch_set in enumerate(sets_of_pitch_values): 
print(f"Set {index + 1}: {pitch_set}") 
Output: 
Modified Sets of Pitch Frequencies: 
Set 1: [153.09387 153.09387 153.09387 153.09387 153.09387 153.09387 
153.09387 
 153.09387 153.09387 153.09387 153.09387 153.09387] 
Set 2: [150.4458 150.4458 150.4458 150.4458 150.4458 150.4458 150.4458 
150.4458 
 150.4458 150.4458 150.4458 150.4458] 
Set 3: [151.03691 151.03691 151.03691 151.03691 151.03691 151.03691 
151.03691 
 151.03691 151.03691 151.03691 151.03691 151.03691] 
Set 4: [158.02992 158.02992 158.02992 158.02992 158.02992 158.02992 
158.02992 
 158.02992 158.02992 158.02992 158.02992 158.02992] 
Set 5: [159.42113 159.42113 159.42113 159.42113 159.42113 159.42113 
159.42113 
 159.42113 159.42113 159.42113 159.42113 159.42113] 
Set 6: [173.8389 173.8389 173.8389 173.8389 173.8389 173.8389 173.8389 
173.8389 
 173.8389 173.8389 173.8389 173.8389] 
Set 7: [180.5283 180.5283 180.5283 180.5283 180.5283 180.5283 180.5283 
180.5283 
 180.5283 180.5283 180.5283 180.5283] 
Set 8: [183.38844 183.38844 183.38844 183.38844 183.38844 183.38844 
183.38844 
 183.38844 183.38844 183.38844 183.38844 183.38844] 
14 
Set 9: [192.83878 192.83878 192.83878 192.83878 192.83878 192.83878 
192.83878 
 192.83878 192.83878 192.83878 192.83878 192.83878] 
Set 10: [201.60442 201.60442 201.60442 201.60442 201.60442 201.60442 
201.60442 
 201.60442 201.60442 201.60442 201.60442 201.60442] 
Set 11: [205.16835 205.16835 205.16835 205.16835 205.16835 205.16835 
205.16835 
 205.16835 205.16835 205.16835 205.16835 205.16835] 
Set 12: [209.75348 209.75348 209.75348 209.75348 209.75348 209.75348 
209.75348 
 209.75348 209.75348 209.75348 209.75348 209.75348] 
Set 13: [224.51552 224.51552 224.51552 224.51552 224.51552 224.51552 
224.51552 
 224.51552 224.51552 224.51552 224.51552 224.51552] 
Set 14: [231.7641 231.7641 231.7641 231.7641 231.7641 231.7641 231.7641 
231.7641 
 231.7641 231.7641 231.7641 231.7641] 
Set 15: [240.84962 240.84962 240.84962 240.84962 240.84962 240.84962 
240.84962 
 240.84962 240.84962 240.84962 240.84962 240.84962] 
Set 16: [249.66884 249.66884 249.66884 249.66884 249.66884 249.66884 
249.66884 
 249.66884 249.66884 249.66884 249.66884 249.66884] 
Set 17: [255.36737 255.36737 255.36737 255.36737 255.36737 255.36737 
255.36737 
 255.36737 255.36737 255.36737 255.36737 255.36737] 
Set 18: [255.46991 255.46991 255.46991 255.46991 255.46991 255.46991 
255.46991 
 255.46991 255.46991 255.46991 255.46991 255.46991] 
Set 19: [270.27475 270.27475 270.27475 270.27475 270.27475 270.27475 
270.27475 
 270.27475 270.27475 270.27475 270.27475 270.27475] 
Set 20: [283.45197 283.45197 283.45197 283.45197 283.45197 283.45197 
283.45197 
 283.45197 283.45197 283.45197 283.45197 283.45197] 
Set 21: [296.0411 296.0411 296.0411 296.0411 296.0411 296.0411 296.0411 
296.0411 
 296.0411 296.0411 296.0411 296.0411] 
Set 22: [300.77466 300.77466 300.77466 300.77466 300.77466 300.77466 
300.77466 
 300.77466 300.77466 300.77466 300.77466 300.77466] 
Set 23: [303.9558 303.9558 303.9558 303.9558 303.9558 303.9558 303.9558 
303.9558 
 303.9558 303.9558 303.9558 303.9558] 
Set 24: [311.48215 311.48215 311.48215 311.48215 311.48215 311.48215 
311.48215 
 311.48215 311.48215 311.48215 311.48215 311.48215] 
Set 25: [312.4271 312.4271 312.4271 312.4271 312.4271 312.4271 312.4271 
312.4271 
 312.4271 312.4271 312.4271 312.4271] 
15 
Set 26: [311.25192 311.25192 311.25192 311.25192 311.25192 311.25192 
311.25192 
 311.25192 311.25192 311.25192 311.25192 311.25192] 
Set 27: [327.70966 327.70966 327.70966 327.70966 327.70966 327.70966 
327.70966 
 327.70966 327.70966 327.70966 327.70966 327.70966] 
Set 28: [350.8226 350.8226 350.8226 350.8226 350.8226 350.8226 350.8226 
350.8226 
 350.8226 350.8226 350.8226 350.8226] 
Set 29: [364.40646 364.40646 364.40646 364.40646 364.40646 364.40646 
364.40646 
 364.40646 364.40646 364.40646 364.40646 364.40646] 
Set 30: [376.04395 376.04395 376.04395 376.04395 376.04395 376.04395 
376.04395 
 376.04395 376.04395 376.04395 376.04395 376.04395] 
Set 31: [378.6707 378.6707 378.6707 378.6707 378.6707 378.6707 378.6707 
378.6707 
 378.6707 378.6707 378.6707 378.6707] 
Set 32: [384.47522 384.47522 384.47522 384.47522 384.47522 384.47522 
384.47522 
 384.47522 384.47522 384.47522 384.47522 384.47522] 
Set 33: [394.64636 394.64636 394.64636 394.64636 394.64636 394.64636 
394.64636 
 394.64636 394.64636 394.64636 394.64636 394.64636] 
Set 34: [415.71277 415.71277 415.71277 415.71277 415.71277 415.71277 
415.71277 
 415.71277 415.71277 415.71277 415.71277 415.71277] 
Set 35: [436.85983 436.85983 436.85983 436.85983 436.85983 436.85983 
436.85983 
 436.85983 436.85983 436.85983 436.85983 436.85983] 
Set 36: [452.17188 452.17188 452.17188 452.17188 452.17188 452.17188 
452.17188 
 452.17188 452.17188 452.17188 452.17188 452.17188] 
Set 37: [459.06442 459.06442 459.06442 459.06442 459.06442 459.06442 
459.06442 
 459.06442 459.06442 459.06442 459.06442 459.06442] 
Set 38: [465.31137 465.31137 465.31137 465.31137 465.31137 465.31137 
465.31137 
 465.31137 465.31137 465.31137 465.31137 465.31137] 
Set 39: [469.81223 469.81223 469.81223 469.81223 469.81223 469.81223 
469.81223 
 469.81223 469.81223 469.81223 469.81223 469.81223] 
Set 40: [491.54803 491.54803 491.54803 491.54803 491.54803 491.54803 
491.54803 
 491.54803 491.54803 491.54803 491.54803 491.54803] 
Set 41: [501.10867 501.10867 501.10867 501.10867 501.10867 501.10867 
501.10867 
 501.10867 501.10867 501.10867 501.10867 501.10867] 
Set 42: [510.4448 510.4448 510.4448 510.4448 510.4448 510.4448 510.4448 
510.4448 
 510.4448 510.4448 510.4448 510.4448] 
16 
Set 43: [538.04846 538.04846 538.04846 538.04846 538.04846 538.04846 
538.04846 
 538.04846 538.04846 538.04846 538.04846 538.04846] 
Set 44: [554.65015 554.65015 554.65015 554.65015 554.65015 554.65015 
554.65015 
 554.65015 554.65015 554.65015 554.65015 554.65015] 
Set 45: [582.3097 582.3097 582.3097 582.3097 582.3097 582.3097 582.3097 
582.3097 
 582.3097 582.3097 582.3097 582.3097] 
Set 46: [607.72156 607.72156 607.72156 607.72156 607.72156 607.72156 
607.72156 
 607.72156 607.72156 607.72156 607.72156 607.72156] 
Set 47: [625.81934 625.81934 625.81934 625.81934 625.81934 625.81934 
625.81934 
 625.81934 625.81934 625.81934 625.81934 625.81934] 
Set 48: [665.9771 665.9771 665.9771 665.9771 665.9771 665.9771 665.9771 
665.9771 
 665.9771 665.9771 665.9771 665.9771] 
Set 49: [722.6002 722.6002 722.6002 722.6002 722.6002 722.6002 722.6002 
722.6002 
 722.6002 722.6002 722.6002 722.6002] 
Set 50: [810.6594 810.6594 810.6594 810.6594 810.6594 810.6594 810.6594 
810.6594 
 810.6594 810.6594 810.6594 810.6594] 
Set 51: [912.174 912.174 912.174 912.174 912.174 912.174 912.174 912.174 
912.174 
 912.174 912.174 912.174] 
Set 52: [1051.3744 1051.3744 1051.3744 1051.3744 1051.3744 1051.3744 
1051.3744 
 1051.3744 1051.3744 1051.3744 1051.3744 1051.3744] 
Set 53: [1283.3243 1283.3243 1283.3243 1283.3243 1283.3243 1283.3243 
1283.3243 
 1283.3243 1283.3243 1283.3243 1283.3243 1283.3243] 
Set 54: [1490.8215 1490.8215 1490.8215 1490.8215 1490.8215 1490.8215 
1490.8215 
 1490.8215 1490.8215 1490.8215 1490.8215 1490.8215] 
Set 55: [1750.5726 1750.5726 1750.5726 1750.5726 1750.5726 1750.5726 
1750.5726 
 1750.5726 1750.5726 1750.5726 1750.5726 1750.5726] 
Set 56: [2022.2607 2022.2607 2022.2607 2022.2607 2022.2607 2022.2607 
2022.2607 
 2022.2607 2022.2607 2022.2607 2022.2607 2022.2607] 
Set 57: [2302.3127 2302.3127 2302.3127 2302.3127 2302.3127 2302.3127 
2302.3127 
 2302.3127 2302.3127 2302.3127 2302.3127 2302.3127] 
Set 58: [2544.9946 2544.9946 2544.9946 2544.9946 2544.9946 2544.9946 
2544.9946 
 2544.9946 2544.9946 2544.9946 2544.9946 2544.9946] 
Set 59: [2816.6455 2816.6455 2816.6455 2816.6455 2816.6455 2816.6455 
2816.6455 
 2816.6455 2816.6455 2816.6455 2816.6455 2816.6455] 
17 
Set 60: [3075.7358 3075.7358 3075.7358 3075.7358 3075.7358 3075.7358 
3075.7358 
 3075.7358 3075.7358 3075.7358 3075.7358 3075.7358] 
Set 61: [3319.4214 3319.4214 3319.4214 3319.4214 3319.4214 3319.4214 
3319.4214 
 3319.4214 3319.4214 3319.4214 3319.4214 3319.4214] 
Set 62: [3539.2173 3539.2173 3539.2173 3539.2173 3539.2173 3539.2173 
3539.2173 
 3539.2173 3539.2173 3539.2173 3539.2173 3539.2173] 
Set 63: [3755.837 3755.837 3755.837 3755.837 3755.837 3755.837 3755.837 
3755.837 
 3755.837 3755.837 3755.837 3755.837] 
Set 64: [3935.5737 3935.5737 3935.5737 3935.5737 3935.5737 3935.5737 
3935.5737 
 3935.5737] 
 Displaying the combined frequencies of all the sets from 12octaves 
import librosa 
import numpy as np 
# Load the audio file
audio_path = r"/content/drive/MyDrive/Humming_Project/Humming test.opus"
y, sr = librosa.load(audio_path) 
# Extract pitch information using Librosa's pitch detection function
pitches, magnitudes = librosa.core.piptrack(y=y, sr=sr) 
# Get the pitch frequencies (non-zero values from the pitches array)
pitch_frequencies = pitches[pitches > 0] 
# Separate pitch values into sets of 12 consecutive values
sets_of_pitch_values = [pitch_frequencies[i:i+12] for i in range(0, 
len(pitch_frequencies), 12)] 
# Define the target pitch values for reassigning
target_pitch_values = [440, 466.16, 493.88, 523.25, 554.37, 587.33, 
622.25, 659.26, 698.46, 739.99, 783.99, 830.61] 
# Iterate through each set, find the mean value, and assign the entire 
set with the mean
for index, pitch_set in enumerate(sets_of_pitch_values): 
 mean_value = np.mean(pitch_set) 
 # Assign the entire set with the mean value
 pitch_set[:] = [mean_value] * len(pitch_set) 
 # Find the nearest value from the target_pitch_values list
 nearest_value = min(target_pitch_values, key=lambda x: abs(x - 
mean_value)) 
 # Assign the entire set with the nearest value from the 
target_pitch_values list
 sets_of_pitch_values[index][:] = [nearest_value] * len(pitch_set) 
18 
# Create an empty list to store the modified pitch values
modified_pitch_values = [] 
# Iterate through each modified pitch set and add its values to the 
modified_pitch_values list
for pitch_set in sets_of_pitch_values: 
 modified_pitch_values.extend(pitch_set) 
# Print the combined modified pitch values
print("Combined Modified Pitch Frequencies:") 
print(modified_pitch_values) 
Output: 
Modified Sets of Pitch Frequencies: 
Set 1: [440. 440. 440. 440. 440. 440. 440. 440. 440. 440. 440. 440.] 
Set 2: [440. 440. 440. 440. 440. 440. 440. 440. 440. 440. 440. 440.] 
Set 3: [440. 440. 440. 440. 440. 440. 440. 440. 440. 440. 440. 440.] 
Set 4: [440. 440. 440. 440. 440. 440. 440. 440. 440. 440. 440. 440.] 
Set 5: [440. 440. 440. 440. 440. 440. 440. 440. 440. 440. 440. 440.] 
Set 6: [440. 440. 440. 440. 440. 440. 440. 440. 440. 440. 440. 440.] 
Set 7: [440. 440. 440. 440. 440. 440. 440. 440. 440. 440. 440. 440.] 
Set 8: [440. 440. 440. 440. 440. 440. 440. 440. 440. 440. 440. 440.] 
Set 9: [440. 440. 440. 440. 440. 440. 440. 440. 440. 440. 440. 440.] 
Set 10: [440. 440. 440. 440. 440. 440. 440. 440. 440. 440. 440. 440.] 
. 
. 
. 
. 
Set 60: [830.61 830.61 830.61 830.61 830.61 830.61 830.61 830.61 830.61 
830.61 
 830.61 830.61] 
Set 61: [830.61 830.61 830.61 830.61 830.61 830.61 830.61 830.61 830.61 
830.61 
 830.61 830.61] 
Set 62: [830.61 830.61 830.61 830.61 830.61 830.61 830.61 830.61 830.61 
830.61 
 830.61 830.61] 
Set 63: [830.61 830.61 830.61 830.61 830.61 830.61 830.61 830.61 830.61 
830.61 
 830.61 830.61] 
Set 64: [830.61 830.61 830.61 830.61 830.61 830.61 830.61 830.61] 
3.2.4. Modifying the audio based on pitches and saving the audio in the system.
import librosa 
import soundfile as sf 
import numpy as np 
import os 
19 
# Load the audio file
audio_path = r"/content/drive/MyDrive/Humming_Project/Humming test.opus"
y, sr = librosa.load(audio_path) 
# Extract pitch information using Librosa's pitch detection function
pitches, magnitudes = librosa.core.piptrack(y=y, sr=sr) 
# Get the pitch frequencies (non-zero values from the pitches array)
pitch_frequencies = pitches[pitches > 0] 
# Separate pitch values into sets of 12 consecutive values
sets_of_pitch_values = [pitch_frequencies[i:i+12] for i in range(0, 
len(pitch_frequencies), 12)] 
# Define the target pitch values for reassigning
target_pitch_values = [440, 466.16, 493.88, 523.25, 554.37, 587.33, 
622.25, 659.26, 698.46, 739.99, 783.99, 830.61] 
# Define the tunes database (mapping target_pitch_values to audio file 
paths)
tunes_database = { 
 440: "/content/drive/MyDrive/Humming_Project/Tunes - 12 octaves 
/A.mp3", 
 466.16: "/content/drive/MyDrive/Humming_Project/Tunes - 12 octaves /A 
sharp #.mp3", 
 493.88: "/content/drive/MyDrive/Humming_Project/Tunes - 12 octaves 
/B.mp3", 
 523.25: "/content/drive/MyDrive/Humming_Project/Tunes - 12 octaves 
/C.mp3", 
 554.37: "/content/drive/MyDrive/Humming_Project/Tunes - 12 octaves /C 
sharp or D - FLat.mp3", 
 587.33: "/content/drive/MyDrive/Humming_Project/Tunes - 12 octaves 
/D.mp3", 
 622.25: "/content/drive/MyDrive/Humming_Project/Tunes - 12 octaves /D 
sharp #.mp3", 
 659.26: "/content/drive/MyDrive/Humming_Project/Tunes - 12 octaves 
/E.mp3", 
 698.46: "/content/drive/MyDrive/Humming_Project/Tunes - 12 octaves 
/F.mp3", 
 733.99: "/content/drive/MyDrive/Humming_Project/Tunes - 12 octaves /F 
Sharp #.mp3", 
 783.99: "/content/drive/MyDrive/Humming_Project/Tunes - 12 octaves 
/G.mp3", 
 830.61: "/content/drive/MyDrive/Humming_Project/Tunes - 12 octaves /G 
Sharp #.mp3", 
 # ... and so on for other target_pitch_values
} 
# Directory to save the modified audio files
20 
output_directory = 
"/content/drive/MyDrive/Humming_Project/Modified_Audio/"
# Ensure the output directory exists, create it if not
os.makedirs(output_directory, exist_ok=True) 
# Process and save audio corresponding to each set value
for index, pitch_set in enumerate(sets_of_pitch_values): 
 mean_value = np.mean(pitch_set) 
 nearest_value = min(target_pitch_values, key=lambda x: abs(x - 
mean_value)) 
 tune_path = tunes_database.get(nearest_value) 
 if tune_path: 
 tune_audio, _ = librosa.load(tune_path, sr=sr) 
 # Modify the audio based on pitch_set and nearest_value if 
necessary
 modified_audio = tune_audio # Placeholder, modify this based on 
your requirements
 # Save the modified audio as a WAV file
 output_filename = os.path.join(output_directory, 
f"modified_audio_set_{index + 1}.wav") 
 sf.write(output_filename, modified_audio, sr) 
 print(f"Modified audio for pitch set {index + 1} saved to: 
{output_filename}") 
 else: 
 print(f"No tune found for pitch set {index + 1} with mean pitch 
{mean_value}") 
Output: 
Modified audio for pitch set 1 saved to: 
/content/drive/MyDrive/Humming_Project/Modified_Audio/modified_audio_set_
1.wav 
Modified audio for pitch set 2 saved to: 
/content/drive/MyDrive/Humming_Project/Modified_Audio/modified_audio_set_
2.wav 
Modified audio for pitch set 3 saved to: 
/content/drive/MyDrive/Humming_Project/Modified_Audio/modified_audio_set_
3.wav 
Modified audio for pitch set 4 saved to: 
/content/drive/MyDrive/Humming_Project/Modified_Audio/modified_audio_set_
4.wav 
Modified audio for pitch set 5 saved to: 
/content/drive/MyDrive/Humming_Project/Modified_Audio/modified_audio_set_
5.wav 
Modified audio for pitch set 6 saved to: 
/content/drive/MyDrive/Humming_Project/Modified_Audio/modified_audio_set_
6.wav 
Modified audio for pitch set 7 saved to: 
/content/drive/MyDrive/Humming_Project/Modified_Audio/modified_audio_set_
7.wav 
Modified audio for pitch set 8 saved to: 
/content/drive/MyDrive/Humming_Project/Modified_Audio/modified_audio_set_
8.wav 
21 
Modified audio for pitch set 9 saved to: 
/content/drive/MyDrive/Humming_Project/Modified_Audio/modified_audio_set_
9.wav 
Modified audio for pitch set 10 saved to: 
/content/drive/MyDrive/Humming_Project/Modified_Audio/modified_audio_set_
10.wav 
Modified audio for pitch set 11 saved to: 
/content/drive/MyDrive/Humming_Project/Modified_Audio/modified_audio_set_
11.wav 
Modified audio for pitch set 12 saved to: 
/content/drive/MyDrive/Humming_Project/Modified_Audio/modified_audio_set_
12.wav 
Modified audio for pitch set 13 saved to: 
/content/drive/MyDrive/Humming_Project/Modified_Audio/modified_audio_set_
13.wav 
Modified audio for pitch set 14 saved to: 
/content/drive/MyDrive/Humming_Project/Modified_Audio/modified_audio_set_
14.wav 
Modified audio for pitch set 15 saved to: 
/content/drive/MyDrive/Humming_Project/Modified_Audio/modified_audio_set_
15.wav 
Modified audio for pitch set 50 saved to: 
/content/drive/MyDrive/Humming_Project/Modified_Audio/modified_audio_set_
50.wav 
Modified audio for pitch set 51 saved to: 
/content/drive/MyDrive/Humming_Project/Modified_Audio/modified_audio_set_
51.wav 
Modified audio for pitch set 52 saved to: 
/content/drive/MyDrive/Humming_Project/Modified_Audio/modified_audio_set_
52.wav 
Modified audio for pitch set 53 saved to: 
/content/drive/MyDrive/Humming_Project/Modified_Audio/modified_audio_set_
53.wav 
Modified audio for pitch set 54 saved to: 
/content/drive/MyDrive/Humming_Project/Modified_Audio/modified_audio_set_
54.wav 
Modified audio for pitch set 55 saved to: 
/content/drive/MyDrive/Humming_Project/Modified_Audio/modified_audio_set_
55.wav 
Modified audio for pitch set 56 saved to: 
/content/drive/MyDrive/Humming_Project/Modified_Audio/modified_audio_set_
56.wav 
Modified audio for pitch set 57 saved to: 
/content/drive/MyDrive/Humming_Project/Modified_Audio/modified_audio_set_
57.wav 
Modified audio for pitch set 58 saved to: 
/content/drive/MyDrive/Humming_Project/Modified_Audio/modified_audio_set_
58.wav 
Modified audio for pitch set 59 saved to: 
/content/drive/MyDrive/Humming_Project/Modified_Audio/modified_audio_set_
59.wav 
Modified audio for pitch set 60 saved to: 
/content/drive/MyDrive/Humming_Project/Modified_Audio/modified_audio_set_
60.wav 
Modified audio for pitch set 61 saved to: 
/content/drive/MyDrive/Humming_Project/Modified_Audio/modified_audio_set_
61.wav 
Modified audio for pitch set 62 saved to: 
/content/drive/MyDrive/Humming_Project/Modified_Audio/modified_audio_set_
62.wav 
22 
Modified audio for pitch set 63 saved to: 
/content/drive/MyDrive/Humming_Project/Modified_Audio/modified_audio_set_
63.wav 
Modified audio for pitch set 64 saved to: 
/content/drive/MyDrive/Humming_Project/Modified_Audio/modified_audi
o_set_64.wav
 Merging all the generated audio into a single file
import librosa 
import soundfile as sf 
import numpy as np 
import os 
def process_audio(audio_path): 
 # Load the audio file
 y, sr = librosa.load(audio_path) 
 # Extract pitch information using Librosa's pitch detection function
 pitches, magnitudes = librosa.core.piptrack(y=y, sr=sr) 
 # Get the pitch frequencies (non-zero values from the pitches array)
 pitch_frequencies = pitches[pitches > 0] 
 # Separate pitch values into sets of 12 consecutive values
 sets_of_pitch_values = [pitch_frequencies[i:i+12] for i in range(0, 
len(pitch_frequencies), 12)] 
 # Define the target pitch values for reassigning
 target_pitch_values = [440, 466.16, 493.88, 523.25, 554.37, 587.33, 
622.25, 659.26, 698.46, 739.99, 783.99, 830.61] 
 # Iterate through each set, find the mean value, and assign the 
entire set with the mean
 for index, pitch_set in enumerate(sets_of_pitch_values): 
 mean_value = np.mean(pitch_set) 
 # Assign the entire set with the mean value
 pitch_set[:] = [mean_value] * len(pitch_set) 
 # Find the nearest value from the target_pitch_values list
 nearest_value = min(target_pitch_values, key=lambda x: abs(x - 
mean_value)) 
 # Assign the entire set with the nearest value from the 
target_pitch_values list
 sets_of_pitch_values[index][:] = [nearest_value] * len(pitch_set) 
 # Create an empty list to store the modified pitch values
 modified_pitch_values = [] 
 # Iterate through each modified pitch set and add its values to the 
modified_pitch_values list
 for pitch_set in sets_of_pitch_values: 
 modified_pitch_values.extend(pitch_set) 
23 
 return modified_pitch_values, sr 
# Define the input audio path
input_audio_path = r"/content/drive/MyDrive/Humming_Project/Humming 
test.opus"
# Process the input audio
modified_pitch_values, sr = process_audio(input_audio_path) 
# Define the tunes database (mapping target_pitch_values to audio file 
paths)
tunes_database = { 
 # ... (your tunes database entries remain unchanged)
} 
# Modify the audio based on modified_pitch_values
modified_audio = np.array([]) 
for pitch_value in modified_pitch_values: 
 tune_path = tunes_database.get(pitch_value) 
 if tune_path: 
 tune_audio, _ = librosa.load(tune_path, sr=sr) 
 # Modify the audio based on pitch_value and tune_audio if 
necessary
 modified_audio = np.concatenate((modified_audio, tune_audio)) # 
Concatenate the modified audio
# Trim the modified audio to match the duration of the input audio
input_duration = librosa.get_duration(y=modified_audio, sr=sr) 
target_duration = librosa.get_duration(y=y, sr=sr) 
modified_audio = modified_audio[:int(target_duration * sr)] # Truncate 
the modified audio
# Save the modified audio as a single WAV file
output_filename = 
"/content/drive/MyDrive/Humming_Project/Modified_Audio/Sample3.wav"
sf.write(output_filename, modified_audio, sr) 
print(f"Modified audio saved to: {output_filename}") 
Output: 
Modified audio saved to: 
/content/drive/MyDrive/Humming_Project/Modified_Audio/Sample3.wav
