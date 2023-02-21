import torch
import numpy as np
import pandas as pd
from utils.CustomAudioDataset import CustomAudioDataset, LogSpectrogramExtractor, MinMaxNormalizer

# the CustomAudioDataset contains the full dataset with barks and non-barks samples, the extract_dataset method below
# extracts samples of the speific 'class_name' from it into a new dataset.

def extract_dataset(duration, extractor, normalizer, class_name, invert_match=False):

    dataset = CustomAudioDataset(duration=duration, extractor=extractor, normalizer=normalizer)
    res = []

    # using loop to iterate through list
    for idx, ele in enumerate(dataset.audio_ds['class_name']):
        if not invert_match:
            if ele == class_name: res.append(idx) 
        else:
            if ele != class_name: 
                res.append(idx)
              
    dataset.audio_ds = dataset.audio_ds.loc[dataset.audio_ds.index[res]]
    dataset.audio_ds.index = dataset.audio_ds.index - len(dataset.audio_ds)
    dataset.audio_ds.index = pd.Index(np.arange(0,len(dataset.audio_ds.index),1), dtype='int64')
    
    return dataset

