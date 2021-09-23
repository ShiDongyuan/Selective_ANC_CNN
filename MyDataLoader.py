import os 

from torch.utils.data import Dataset
import pandas as pd 
import torchaudio
import torch

import matplotlib
import matplotlib.pyplot as plt

#------------------------------------------------------------------------
# Function    : plot_specgram()
# Description : Drawing the specgram of the waveform 
#------------------------------------------------------------------------
def plot_specgram(waveform, sample_rate, title="Spectrogram", xlim=None):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].specgram(waveform[c], Fs=sample_rate)
        if num_channels > 1:
            axes[c].set_ylabel(f'Channel {c+1}')
        if xlim:
            axes[c].set_xlim(xlim)
    figure.suptitle(title)
    plt.show(block=False)
#------------------------------------------------------------------------
# Function    : print_stats()
# Description : Drawing the specgram of the waveform 
#------------------------------------------------------------------------
def print_stats(waveform, sample_rate=None, src=None):
    if src:
        print("-" * 10)
        print("Source:", src)
        print("-" * 10)
    if sample_rate:
        print("Sample Rate:", sample_rate)
    print("Shape:", tuple(waveform.shape))
    print("Dtype:", waveform.dtype)
    print(f" - Max:     {waveform.max().item():6.3f}")
    print(f" - Min:     {waveform.min().item():6.3f}")
    print(f" - Mean:    {waveform.mean().item():6.3f}")
    print(f" - Std Dev: {waveform.std().item():6.3f}")
    print()
    print(waveform)
    print()
#------------------------------------------------------------------------
# Class       : minmaxscaler()
# Description : Shrink the data 
#------------------------------------------------------------------------
def minmaxscaler(data):
    min = data.min()
    max = data.max()    
    return (data)/(max-min)

#------------------------------------------------------------------------
# Class       : MyNoiseDataset()
# Description : Build the user's the data set 
#------------------------------------------------------------------------
class MyNoiseDataset(Dataset):

    def __init__(self, folder, annotations_file):
        self.folder = folder 
        self.annotations_file = pd.read_csv(os.path.join(folder, annotations_file))
    
    def __len__(self):
        return len(self.annotations_file)

    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        label             = self._get_audio_sample_label(index)
        signal,_          = torchaudio.load(os.path.join(self.folder, audio_sample_path))
        signal            = minmaxscaler(signal)
        return signal, label
    
    def _get_audio_sample_path(self, index):
        path = self.annotations_file.iloc[index, 1]
        return path

    def _get_audio_sample_label(self, index):
        label = self.annotations_file.iloc[index,2]
        return label 
    
if __name__ == "__main__":
    ANNOTATIONS_FILE = "DATA_1\Index.csv"

    #pd.read_csv("DATA_1\Index.csv")
    Noise_Data_set = MyNoiseDataset(ANNOTATIONS_FILE)

    signal, label = Noise_Data_set[3024]
    singal_shrink = minmaxscaler(signal)

    plot_specgram(signal, 16000)
    print_stats(signal,1600)
    print(signal.shape)
    print(label)
    plot_specgram(singal_shrink, 16000)
    print_stats(singal_shrink,1600)
    print(f"The number of samples in the Dataset is: {Noise_Data_set.__len__()}")

    i = 5

        
    

        

        