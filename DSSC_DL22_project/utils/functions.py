from sklearn.model_selection import train_test_split
from torch.utils.data import Subset


def train_test_dataset(dataset, test_split=0.25):
    train_idx, test_idx = train_test_split(list(range(len(dataset))), test_size=test_split, shuffle=True)
    datasets = {}
    datasets['train'] = Subset(dataset, train_idx)
    datasets['test'] = Subset(dataset, test_idx)
    return datasets


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
    

from matplotlib import pyplot as plt


def set_default(figsize=(10, 10), dpi=100):
    plt.style.use(['dark_background', 'bmh'])
    plt.rc('axes', facecolor='k')
    plt.rc('figure', facecolor='k')
    plt.rc('figure', figsize=figsize, dpi=dpi)


def plot_data(X, y, d=0, auto=False, zoom=1):
    X = X.cpu()
    y = y.cpu()
    plt.scatter(X.numpy()[:, 0], X.numpy()[:, 1], c=y, s=20, cmap=plt.cm.Spectral)
    plt.axis('square')
    axis_range = torch.min(torch.max(X[:,0]), torch.max(X[:,1]))
    plt.axis(np.array((-(axis_range * 3)/10, axis_range * 3, -(axis_range * 1.1)/10, axis_range * 1.1)) * zoom)
    if auto is True: plt.axis('equal')
    plt.axis('off')

    _m, _c = 0, '.15'
    plt.axvline(0, ymin=_m, color=_c, lw=1, zorder=0)
    plt.axhline(0, xmin=_m, color=_c, lw=1, zorder=0)

    
def plot_model(X, y, model):
    model.cpu()
    X = X.cpu()
    y = y.cpu()
    axis_ref_val = torch.min(torch.max(X[:,0]), torch.max(X[:,1])).item()
    #mesh = np.arange(0, axis_ref_val * 3, 1)
    mesh_x = np.arange(0, torch.max(X[:,0]) * 1.1, torch.max(X[:,0]) / 100)
    mesh_y = np.arange(0, torch.max(X[:,1]) * 1.1, torch.max(X[:,1]) / 100)
    #xx, yy = np.meshgrid(mesh, mesh)
    xx, yy = np.meshgrid(mesh_x, mesh_y)
    with torch.no_grad():
        data = torch.from_numpy(np.vstack((xx.reshape(-1), yy.reshape(-1))).T).float()
        Z = model(data).detach()
    Z = np.round(Z).reshape(xx.shape, yy.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.3)
    #plot_data(X, y)
    plt.scatter(X.numpy()[:, 0], X.numpy()[:, 1], c=y, s=20, cmap=plt.cm.Spectral)


from matplotlib.ticker import MaxNLocator
import pickle
import math

def plot_loss(y_loss, RL_weight):
    
    X = np.arange(1, len(y_loss['train']) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle("Losses for RL weight = " + "{:.0e}".format(RL_weight))
    fig.tight_layout(pad=1.0)
    fig.set_dpi(150)
    ax1.plot(X, [math.log(i,10) for i in y_loss['train']], color='r', label='train')
    ax1.plot(X, [math.log(i,10) for i in y_loss['test']], color='b', label='test')
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.set_title('Train and test losses')
    ax1.legend()
    ax2.plot(X, [math.log(i,10) for i in y_loss['train_RL']], color='y', label='reconstruction RL')
    ax2.plot(X, [math.log(i,10) for i in y_loss['train_KL']], color='g', label='regularization KL')
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax2.set_title('Reconstruction and regularization')
    ax2.legend()
    fig.supxlabel('Epoch')
    fig.supylabel('Log loss')

    
def save_loss(y_loss, device, RL_weight):
    # save loss dictionary to .pkl file with used RL_weight as suffix
    loss_file = 'loss_RL_weight_' + str(device) + '_{:.0e}'.format(RL_weight) + '.pkl'
    PATH = os.path.join(os.getcwd(), "saved_losses/" + loss_file)
    with open(PATH, 'wb') as fp:
        pickle.dump(y_loss, fp)
        print('dictionary saved successfully to file ' + loss_file)

        
def load_loss(device, RL_weight):
    # load loss dictionary from .pkl file with used RL_weight as suffix
    loss_file = 'loss_RL_weight_' + str(device) + '_{:.0e}'.format(RL_weight) + '.pkl'
    PATH = os.path.join(os.getcwd(), "saved_losses/" + loss_file)
    with open(PATH, 'rb') as fp:
        data = fp.read()
    loss_data = pickle.loads(data)
    print('dictionary imported successfully from file ' + loss_file)
    
    return loss_data

        
import os
        

def save_model(audio_folder, model, device, RL_weight):
    # save model parameters as per official documentation 
    # https://pytorch.org/tutorials/beginner/saving_loading_models.html
    # state_dict is simply a Python dictionary object that maps each layer to its parameters tensor
    FILENAME = audio_folder + '_' + str(device) + '_{:.0e}'.format(RL_weight) + '.pth'
    PATH = os.path.join(os.getcwd(), "saved_models/" + FILENAME)
    # Save computed model in 'saved_models' directory
    torch.save(model.state_dict(), PATH)
    print('model successfully saved to file ' + FILENAME)
    
    
from utils.VariationalAutoEncoder import VAE


def load_model(audio_folder, input_shape, conv_filters, conv_kernels, conv_strides, latent_space_dim, device, RL_weight):
    # Load a previously computed model from 'saved_models' directory
    FILENAME = audio_folder + '_' + str(device) + '_' + '{:.0e}'.format(RL_weight) + '.pth'
    PATH = os.path.join(os.getcwd(), "saved_models/" + FILENAME)
    
    model = VAE(input_shape,
              conv_filters,
              conv_kernels,
              conv_strides,
              latent_space_dim)

    model.load_state_dict(torch.load(PATH))

    model.to(device)
    model.eval()
    print('model successfully loaded from file ' + FILENAME)
    
    return model


import librosa
from fastaudio.core.all import *
import soundfile as sf


#function that computes the time-domain audio waveform given the Mel-scaled log spectrogram
def logspec_to_audio(logspec, hop_length):
    # reshape the log spectrogram
    logspec = logspec[0, 0, :, :]
    # log spectrogram -> spectrogram
    spec = librosa.db_to_power(logspec, ref=1.0)
    # from melspec to audio
    signal = librosa.feature.inverse.mel_to_audio(spec, hop_length=hop_length)
    
    return signal


# plot audio waveform and a snippet that lets you listen to the file
def plot_audio(folder, filename, audio_samples, sample_rate):
    wav_path = os.path.join(folder, filename)
    # write audio time samples
    sf.write(wav_path, audio_samples, samplerate=sample_rate)
    # plot
    AudioTensor.create(wav_path).show()

    
# given a set of Mel-scaled log spectrograms stacked into a numpy array, plot them   
def plot_logspec(logspec_array):
    N = logspec_array.shape[0]
    dim = math.ceil(np.sqrt(N))
    
    vmin = np.min(np.array(logspec_array))
    vmax = np.max(np.array(logspec_array))

    fig = plt.figure()

    i = 1
    for logspec in logspec_array:
        logspec = logspec[0, :, :]
        ax = fig.add_subplot(dim,dim,i)
        mesh = ax.pcolormesh(logspec)
        mesh.set_clim(vmin,vmax)
        # Visualizing colorbar part -start
        fig.colorbar(mesh,ax=ax)
        fig.tight_layout()
        # Visualizing colorbar part -end
        i += 1 


# computes the reconstuction (RL) and the regularization (KL) losses for an input audio dataset
# outputs three nunpy arrays with RL, KL and class name label for each audio sample in input dataset
def compute_RL_KL(model, device, normalizer, audio_dataset):
    RL = []
    KL = []
    labels = []
    
    i = 0
    for (logspec_norm, min_val, max_val, class_name) in zip(audio_dataset['logspec'],
                                                audio_dataset['min'],
                                                audio_dataset['max'],
                                                audio_dataset['class_name']):

        #encode
        mu_logvar = model.encoder(torch.tensor(logspec_norm).unsqueeze(0).unsqueeze(0).to(device))
        mu_logvar = mu_logvar.view(-1, 2, model.latent_space_dim)
        mu = mu_logvar[:, 0, :]
        logvar = mu_logvar[:, 1, :]

        #decode
        x_hat_norm = model.decoder(mu).cpu().detach()

        Recon_Loss = nn.MSELoss()
        RL.append(Recon_Loss(x_hat_norm, torch.tensor(logspec_norm)))
        KL.append(0.5 * torch.sum(logvar.exp() - logvar - 1 + mu.pow(2)))
        labels.append(class_name)
        i += 1
        
    KL = torch.tensor(KL).cpu().detach().numpy()
    RL = torch.tensor(RL).cpu().detach().numpy()
    
    return RL, KL, mu, logvar, labels

# plot reconstruction vs regularization loss for a 
def plot_RL_KL(test_ds, RL_weight):
    plt.figure(1)
    plt.title("VAE KL loss vs RL loss for RL_weight={:.0e}".format(RL_weight))
    plt.xlabel("Regularization loss - KL")
    plt.ylabel("Reconstruction loss - RL")

    for i in range(len(test_ds)):
        if test_ds["class"][i] == "bark":
            plt.scatter(test_ds["KL"][i], test_ds["RL"][i], color = 'blue')
        elif test_ds["class"][i] == "dog_bark":
            plt.scatter(test_ds["KL"][i], test_ds["RL"][i], color = 'red')
        else:
            plt.scatter(test_ds["KL"][i], test_ds["RL"][i], color = 'orange')
    
    plt.show()