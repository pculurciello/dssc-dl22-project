import os
import random
import pandas as pd
import numpy as np
import torch
import librosa
from torch.utils.data import Dataset
from os.path import exists
from utils.write_csv import write_csv

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


class CustomAudioDataset(Dataset):
    def __init__(self, audio_folder, transform=None, target_transform=None):
        self.audio_dir = os.path.join(os.getcwd(), audio_folder)
        # generate annotations CSV file
        self.class_map = {"not_bark" : 0, "bark": 1}
        self.normalizer = None
        self.extractor = None
        self.transform = transform
        self.target_transform = target_transform
                 
    def _gen_annotations_file(self):
        annotations_file = self.audio_dir + '.csv'
        if not exists(annotations_file):
            write_csv(self.audio_dir, annotations_file)
        else:
            print("Annotations file already exists: " + annotations_file) 
    
    def _load_dataset(self):
        annotations_file = self.audio_dir + '.csv'
        pickle_file = self.audio_dir + '.pkl'
        if exists(pickle_file):
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
        label = self.audio_ds.iloc[idx, 1]
        class_id = self.class_map[label]
        audio_tensor = torch.from_numpy(audio).unsqueeze(0)
        class_id = torch.tensor([class_id])
        if self.transform:
            audio = self.transform(audio)
        if self.target_transform:
            label = self.target_transform(label)         
        return audio_tensor, min_val, max_val, class_id
    
    def _add_logspec(self, dataset):
        df = dataset
        new_column = [None] * len(df)
        df['logspec'] = df['min'] = df['max'] = new_column      
        for idx in range(len(df)):
            audio_path = os.path.join(self.audio_dir, df.iloc[idx]['file_name'])
            audio, sample_rate = librosa.load(audio_path, res_type='kaiser_fast')
            feature = self.extractor.extract(audio)
            norm_feature = self.normalizer.normalize(feature)
            df['logspec'][idx] = norm_feature       
            df['min'][idx] = feature.min()
            df['max'][idx] = feature.max()
        return df