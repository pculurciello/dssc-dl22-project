import os
import random
import pandas as pd
import numpy as np
import torch
import librosa
from torch.utils.data import Dataset
from os.path import exists
from utils.write_csv import write_csv

class CustomAudioDataset(Dataset):
    def __init__(self, audio_folder, normalize=False, transform=None, target_transform=None):
        self.annotations_file = os.path.join(os.getcwd(), audio_folder + '.csv')
        self.pickle_file = os.path.join(os.getcwd(), audio_folder + '.pkl')
        self.audio_dir = os.path.join(os.getcwd(), audio_folder)
        # generate annotations CSV file
        self._gen_annotations_file()
        self.class_map = {"not_bark" : 0, "bark": 1}
        self.normalize = normalize
        self.transform = transform
        self.target_transform = target_transform
        self.audio_ds = self._load_dataset()
        self.audio_ds = self._add_melspec()
        self.melspec_mean, self.melspec_std = self._compute_mean_std()
        
        if self.normalize:
            self.audio_ds = self._normalize_dataset()
                 
    def _gen_annotations_file(self):    
        if not exists(self.audio_dir + '.csv'):
            write_csv(self.audio_dir, self.annotations_file)
        else:
            print("Annotations file already exists: " + self.annotations_file) 
    
    def _load_dataset(self):
        # load the unnormalized version of melspecs
        if exists(self.pickle_file):
            df = pd.read_pickle(self.pickle_file)
        else:
            df = pd.read_csv(self.annotations_file)
            
        return df
        
    def _save_dataset(self):
        #print('audio_ds at the beginning of _save_dataset()')
        #print(self.audio_ds['melspec'])
        if self.normalize:
            # save the unnormalized version of melspecs
            self.audio_ds = self._unnormalize_dataset()
            self.audio_ds.to_pickle(self.pickle_file)
            self.audio_ds = self._normalize_dataset()
        else: 
             # melspecs are already unnormalized
            self.audio_ds.to_pickle(self.pickle_file)
            
        #print('audio_ds at the end of _save_dataset()')
        #print(self.audio_ds['melspec'])
        
    def __len__(self):
        return len(self.audio_ds)

    def __getitem__(self, idx):
        audio = self.audio_ds['melspec'][idx]
        label = self.audio_ds.iloc[idx, 1]
        class_id = self.class_map[label]
        audio_tensor = torch.from_numpy(audio).unsqueeze(0)
        class_id = torch.tensor([class_id])
        if self.transform:
            audio = self.transform(audio)
        if self.target_transform:
            label = self.target_transform(label)
          
        return audio_tensor, class_id
    
    def _add_melspec(self):
        df = self.audio_ds
        
        if not exists(self.pickle_file):
            melspec_column = [None] * len(self.audio_ds)
            df['melspec'] = melspec_column      
            for idx in range(len(df)):
                audio_path = os.path.join(self.audio_dir, df.iloc[idx]['file_name'])
                audio, sample_rate = librosa.load(audio_path, res_type='kaiser_fast')
                ## normalize audio
                #audio_norm = (audio - np.mean(audio)) / np.std(audio)
                #melspec = librosa.feature.melspectrogram(y=audio_norm, sr=sample_rate)
                melspec = librosa.feature.melspectrogram(y=audio, sr=sample_rate)
                df['melspec'][idx] = melspec
           
        return df 
         
    def _normalize_dataset(self):
        df = self.audio_ds
        melspec = torch.tensor(df['melspec'])
        
        #print(df['melspec'])
        #print(melspec.shape[0])
        #print(self.melspec_mean)
        #print(self.melspec_std)
        #print('- - - - - - - - -')
        
        for i in range(melspec.shape[0]):
            df['melspec'][i] = ((melspec[i] - self.melspec_mean) / self.melspec_std).numpy()

        #print(df['melspec'])    
        #print(torch.tensor(df['melspec']).mean())
        #print(torch.tensor(df['melspec']).std())
        #print('- - - - - - - - -')
        
        return df
        
    def _unnormalize_dataset(self):
        #print('audio_ds at the beginning of _unnormalize_dataset()')
        #print(self.audio_ds['melspec'])
        df = self.audio_ds
        melspec = torch.tensor(df['melspec'])
        for i in range(melspec.shape[0]):
            df['melspec'][i] = ((melspec[i] * self.melspec_std) + self.melspec_mean).numpy()
            
        #print('audio_ds at the end of _unnormalize_dataset()')
        #print(self.audio_ds['melspec'])
        
        return df
            
    def _compute_mean_std(self):
        melspec = torch.tensor(self.audio_ds['melspec'])
        melspec_mean = melspec.mean()
        melspec_std = melspec.std()
        
        return melspec_mean, melspec_std