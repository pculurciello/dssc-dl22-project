import librosa
import soundfile as sf
import os
from matplotlib import pyplot as plt
import math
import numpy as np
 
- - - - 

utils_path = os.getcwd()
parent_path = os.path.abspath(os.path.join(utils_path, os.pardir))
audio_path = os.path.join(parent_path, 'audio')

- - - - 

def plot_audio(filename):
    audio_path = os.path.join(barks_path, filename)
    audio, sr = librosa.load(audio_path, res_type='kaiser_fast')
#     fig = plt.figure(figsize=(50,50))
#     ax.plot(audio)
#     ax.title.set_text(filename.split("_")[0:2])
    
    plt.figure(1)
    plt.title(filename.split("_")[0:2])
    plt.plot(audio)
    plt.show()

- - - - -

barks_file = 'barks_capture_20221014.wav'
barks_folder = 'barks_2s'

barks_capture = os.path.join(audio_path, barks_file)
barks_path = os.path.join(parent_path, barks_folder)
duration = 2

# First load the file
audio, sr = librosa.load(barks_capture)

# Get number of samples for duration seconds
buffer = duration * sr

samples_total = len(audio)
samples_wrote = 0
counter = 1

while samples_wrote < samples_total:

    # check if the buffer is not exceeding total samples 
    if buffer > (samples_total - samples_wrote):
        buffer = samples_total - samples_wrote

    block = audio[samples_wrote : (samples_wrote + buffer)]
    out_filename = os.path.join(barks_path, "split_" + str(counter) + "_" + barks_file)
    #  Write duration second segment
    sf.write(out_filename, block, samplerate=sr)

    counter += 1
    samples_wrote += buffer

- - - - 

file_count = 0
for file in os.listdir(barks_path):
    audio_path = os.path.join(barks_path, file)
    # get number of audio files in dir
    if os.path.isfile(audio_path):
        file_count += 1 

dim = math.ceil(np.sqrt(file_count))

columns = 10
rows = math.ceil(file_count / columns)

fig = plt.figure(figsize=(50,50))

i = 1
for file in os.listdir(barks_path):
    audio_path = os.path.join(barks_path, file)
    audio, sr = librosa.load(audio_path, res_type='kaiser_fast')
    delete = True
    if np.max(np.abs(audio)) > 0.1:
        # if files starts or ends with sound (first 10 msecs) it may be trimmed after the split, so delete the chunck
        if np.mean(np.abs(audio[:round(sr * 1e-2)])) >= 5e-2 or np.mean(np.abs(audio[-round(sr * 1e-2):])) >= 5e-2:
            print('file ' + file + ' contains invalid barks!')
            print(np.mean(np.abs(audio[:round(sr * 1e-2)])))
            print(np.mean(np.abs(audio[-round(sr * 1e-2):])))
        else:
            print('file ' + file + ' contains valid barks!')
            ax = fig.add_subplot(rows,columns,i)
            ax.plot(audio)
            ax.title.set_text(file.split("_")[0:2])
            delete = False
    else:
        print('file ' + file + ' contains noise only!')
    if delete: os.remove(audio_path) 
    i += 1 

- - - - 

# rename files
i = 1
for file in os.listdir(barks_path):
    src_path = os.path.join(barks_path, file)
    dst_path = os.path.join(barks_path, 'barks_0' + str(i) + '.wav' if i < 10 else 'barks_' + str(i) + '.wav')
    os.rename(src_path, dst_path)
    i += 1 

- - - - 

plot_audio("barks_224.wav")

- - - - 