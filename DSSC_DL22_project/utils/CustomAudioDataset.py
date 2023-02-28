import os
import pandas as pd
import librosa
import numpy as np
import torch
from torch.utils.data import Dataset
from utils.write_csv import write_csv

# compute the Mel-scaled log spectrogram for an input audio signal
class LogSpectrogramExtractor:
    """LogSpectrogramExtractor extracts melspectrogram (in dB) from a
    time-series signal.
    """
    
    def __init__(self, n_mels, hop_length, sample_rate=22050):
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.sample_rate = sample_rate

    def extract(self, audio):
        melspec = librosa.feature.melspectrogram(audio, 
                                                 n_mels=self.n_mels, 
                                                 hop_length=self.hop_length, 
                                                 sr=self.sample_rate)       
        # Convert to log scale (dB).
        log_melspec = librosa.power_to_db(melspec, ref=1.0)
        
        return log_melspec
    
# Normalize an input vector using the Min-Max algorithm
class MinMaxNormalizer:
    """MinMaxNormaliser applies min max normalisation to an array."""

    def __init__(self, min_val, max_val):
        self.min = min_val
        self.max = max_val

    def normalize(self, array):
        norm_array = (array - array.min()) / (array.max() - array.min())
        norm_array = norm_array * (self.max - self.min) + self.min
        return norm_array

    def denormalize(self, norm_array, original_min, original_max):
        array = (norm_array - self.min) / (self.max - self.min)
        array = array * (original_max - original_min) + original_min
        return array

# Extends the dataset class and creates a dataset structure where the audio signal Mel-Spectrogram, 
# its Min and Max values and the audio signal class are stored in the dataset.audio_ds dataframe.
# Write_csv method mixes the bark samples in audio_dir with all the audio samples from the ESC-10 dataset
# So the dataset.audio_ds dataframe contains the full dataset with barks and non-barks samples
class CustomAudioDataset(Dataset):
    def __init__(self, duration, normalizer = None, extractor = None, transform=None, target_transform=None):
        super().__init__()
        self.duration = duration
        self.audio_folder = 'barks_' + str(duration) + 's'
        self.audio_dir = os.path.join(os.getcwd(), self.audio_folder)
        # class map for binary classification (bark vs non-bark)
        self.class_map = {"not_bark" : 0, "bark": 1}
        self.normalizer = normalizer
        self.extractor = extractor
        self.transform = transform
        self.target_transform = target_transform
        self.audio_ds = self._load_dataset()    
        self._save_dataset()
    
    def _load_dataset(self):
        annotations_file = self.audio_dir + '.csv'
        if not os.path.exists(annotations_file):
            write_csv(self.audio_dir, annotations_file)
        pickle_file = self.audio_dir + '.pkl'
        if os.path.exists(pickle_file):
            df = pd.read_pickle(pickle_file)
        else:
            df = pd.read_csv(annotations_file)
            df = self._add_logspec(df)
        return df
        
    def _save_dataset(self):
        pickle_file = self.audio_dir + '.pkl'
        self.audio_ds.to_pickle(pickle_file)
        
    def __len__(self):
        return len(self.audio_ds)

    def __getitem__(self, idx):
        audio = self.audio_ds['logspec'][idx]
        min_val = self.audio_ds['min'][idx]
        max_val = self.audio_ds['max'][idx]
        class_name = self.audio_ds['class_name'][idx]
        if class_name == 'bark': 
            class_id = self.class_map[class_name]
        else:
            class_id = self.class_map['not_bark']
        audio_tensor = torch.from_numpy(audio).unsqueeze(0)
        class_id = torch.tensor([class_id])
        if self.transform:
            audio = self.transform(audio)
        if self.target_transform:
            label = self.target_transform(label)         
        return audio_tensor, min_val, max_val, class_id, class_name
    
    def _load_audio(self, file_name, class_name):
        if class_name == 'bark':
            audio_path = os.path.join(self.audio_dir, file_name)
        else:
            audio_path = os.path.join(os.getcwd(), 'ESC-10', class_name, file_name)
        return librosa.load(audio_path, res_type='kaiser_fast')
    
    def _add_logspec(self, dataset):
        df = dataset
        new_column = [None] * len(df)
        df['logspec'] = df['min'] = df['max'] = new_column      
        for idx in range(len(df)):
            audio, sample_rate = self._load_audio(df.iloc[idx]['file_name'], df.iloc[idx]['class_name'])
            tot_duration = round(len(audio) / sample_rate)
            frame_start = round((tot_duration - self.duration) / 2 * sample_rate)
            frame_end = frame_start + self.duration * sample_rate
            feature = self.extractor.extract(audio[frame_start:frame_end])
            norm_feature = self.normalizer.normalize(feature)
            df['logspec'][idx] = norm_feature       
            df['min'][idx] = feature.min()
            df['max'][idx] = feature.max()
        return df