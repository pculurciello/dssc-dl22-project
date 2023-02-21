import os
import pandas as pd
import numpy as np
import torch
import librosa
from torch.utils.data import Dataset

class CustomAudioDataset(Dataset):
    def __init__(self, annotations_file, audio_dir, normalize=False, transform=None, target_transform=None):
        self.audio_labels = pd.read_csv(annotations_file)
        self.audio_dir = audio_dir
        self.class_map = {"not_bark" : 0, "bark": 1}
        self.normalize = normalize
        self.transform = transform
        self.target_transform = target_transform
        
        if self.normalize:
            self.mean, self.std = self.compute_mean_std()

    def __len__(self):
        return len(self.audio_labels)

    def __getitem__(self, idx):
        audio_path = os.path.join(self.audio_dir, self.audio_labels.iloc[idx, 0])
        audio = self.extract_mfccs(audio_path)
        label = self.audio_labels.iloc[idx, 1]
        class_id = self.class_map[label]
        audio_tensor = torch.from_numpy(audio)
        class_id = torch.tensor([class_id])
        if self.transform:
            audio = self.transform(audio)
        if self.target_transform:
            label = self.target_transform(label)
        return audio_tensor, class_id
        
    def extract_mfccs(self, file_path):
        try:
            audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
            mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
            if self.normalize:
                for mfcc in range(mfccs.shape[0]):
                    mfccs[mfcc, :] = (mfccs[mfcc, :] - self.mean[mfcc]) / self.std[mfcc]
        except Exception as e:
            print("file could not be loaded: ", file_path, "- - - Error: ", e)
            return None
        return np.expand_dims(mfccs, axis=0)
    
    def compute_mean_std(self):
        # get number of files in audio_dir
        n_files = len([name for name in os.listdir(self.audio_dir) if os.path.isfile(os.path.join(self.audio_dir, name))])
        # get mfccs shape computing mfcc for the first file
        for file in os.listdir(self.audio_dir):
            audio_path = os.path.join(self.audio_dir, file)
            audio, sample_rate = librosa.load(audio_path, res_type='kaiser_fast')
            mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
            n_mfcc = mfccs.shape[0]
            n_samples =  mfccs.shape[1]
            break
        
        # initialize mfccs dataset
        mfccs_ds = np.empty([n_files, n_mfcc, n_samples])
        
        i = 0
        for file in os.listdir(self.audio_dir):
            audio_path = os.path.join(self.audio_dir, file)
            audio, sample_rate = librosa.load(audio_path, res_type='kaiser_fast')
            mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
            mfccs_ds[i] = mfccs
            i += 1
            
        mfccs_mean = np.mean(mfccs_ds,axis=(0,2))
        mfccs_std = np.std(mfccs_ds,axis=(0,2))
     
        return mfccs_mean, mfccs_std
