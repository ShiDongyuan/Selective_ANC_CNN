import torch 
from torch import nn
from torchsummary import summary
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T

from torch.utils.data import DataLoader 
from MyDataLoader import MyNoiseDataset

from ONED_CNN_PRE import OneD_CNN_Predictor
from ONED_CNN_LMSoftmax_PRE_v1 import ONE_CNN_LMSoftmax_Predictor

import numpy as np
import scipy.signal as signal
import matplotlib
import matplotlib.pyplot as plt

from scipy.fft import fft, fftfreq, ifft
import math

from Fixed_Filter_noise_cancellation import Fxied_filters
#-----------------------------------------------------------------------------------
# Function    : additional_noise()
# Description : The additional noise generation 
#-----------------------------------------------------------------------------------
def additional_noise(signal, snr_db):
    signal_power     = signal.norm(p=2)
    # print(signal.shape)
    length           = signal.shape[1]
    additional_noise = np.random.randn(length)
    additional_noise = torch.from_numpy(additional_noise).type(torch.float32).unsqueeze(0)
    noise_power      = additional_noise.norm(p=2)
    snr              = math.exp(snr_db / 10)
    scale            = snr * noise_power / signal_power
    noisy_signal     = (scale * signal + additional_noise) / 2
    return noisy_signal

#-----------------------------------------------------------------------------------
# Function    : BandlimitedNoise_generation_tensor
# Description : The function is used to generate the broadband noise as the design.
#-----------------------------------------------------------------------------------
def BandlimitedNoise_generation_tensor(f_star, Bandwidth, fs, N, SNR):
    # f_star indecats the start of frequency band (Hz)
    # Bandwith denots the bandwith of the boradabnd noise 
    # fs denots the sample frequecy (Hz)
    # N represents the number of point
    len_f = 1024 
    f_end = f_star + Bandwidth
    b2    = signal.firwin(len_f, [f_star, f_end], pass_zero='bandpass', window ='hamming',fs=fs)
    xin   = np.random.randn(N)
    Re    = signal.lfilter(b2,1,xin)
    Noise = Re[len_f-1:]
    Noise = Noise/np.sqrt(np.var(Noise))
    Noise = torch.from_numpy(Noise).type(torch.float32).unsqueeze(0)
    Noise = additional_noise(Noise,SNR)
    #----------------------------------------------------
    return Noise

#-----------------------------------------------------------------------------------
# Function    : plot_specgram()
# Description : This function is used to plot the power spectogram.
#-----------------------------------------------------------------------------------
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

#-----------------------------------------------------------------------------------
# Function    : plot_waveform()
# Description : The function is used to plot the waveform.
#-----------------------------------------------------------------------------------
def plot_waveform(waveform, sample_rate, title="Waveform", xlim=None, ylim=None):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].plot(time_axis, waveform[c], linewidth=1)
        axes[c].grid(True)
        if num_channels > 1:
            axes[c].set_ylabel(f'Channel {c+1}')
        if xlim:
            axes[c].set_xlim(xlim)
        if ylim:
            axes[c].set_ylim(ylim)
    figure.suptitle(title)
    plt.show(block=False)

#-----------------------------------------------------------------------------------
# Function    : plot_waveform()
# Description : The function is used to plot the waveform.
#-----------------------------------------------------------------------------------
class Filter_ID_predictor(OneD_CNN_Predictor,Fxied_filters):

    def __init__(self, MODEL_PATH, MATFILE_PATH, fs,device):
        OneD_CNN_Predictor.__init__(self, MODEL_PATH,device)
        Fxied_filters.__init__(self, MATFILE_PATH, fs)
    
    def predic_ID(self, noise_1):
        similarity_rato = []
        for ii in range(self.len):
            similarity_rato.append(self.cosSimilarity(noise_1, self.Charactors[ii]))
        index = np.argmax(similarity_rato)
        return index

#-----------------------------------------------------------------------------------
# Class :   Filter_ID_predictor_from_1DCNN_LMSoftmax()
# Description:  This class uses the 1DCNN_LMSoftmax model to otbatin the filter ID.
#-----------------------------------------------------------------------------------
class Filter_ID_predictor_from_1DCNN_LMSoftmax(ONE_CNN_LMSoftmax_Predictor, Fxied_filters):
    
    def __init__(self, MODEL_PATH, MATFILE_PATH, fs, Wc, device):
        """
        This is the filter ID predictor, which can predict the index of the pre-trained control filters.
        Parameters:
            MODEL_PATH   - the pre-trained 1DCNN model.
            MATFILE_PATH - the path of the pre-trained control filters.
            fs           - the system sampling rate. 
            Wc           - the parameter of the LMSoftmax layer is [embedding_size x num_classes] Tensor.
            device       - 'cpu' or 'cuda'.
        """
        ONE_CNN_LMSoftmax_Predictor.__init__(self, MODEL_PATH, Wc, device)
        Fxied_filters.__init__(self, MATFILE_PATH, fs)
    
    def predic_ID(self, noise_1):
        """
        This program is used to predict the index of the control filters.
        :param noise: The primary noise has the dimension of [1 x fs samples].
        """
        similarity_rato = []
        for ii in range(self.len):
            similarity_rato.append(self.cosSimilarity(noise_1, self.Charactors[ii]))
        index = np.argmax(similarity_rato)
        return index
               
#------------------------>
#------> main() <--------
#------------------------>
if __name__ == "__main__":
    fs           = 16000
    MATFILE_PATH = 'Pre-train Control filter.mat'
    FILE_NAME_PATH = "Bandlimited_filter.mat"
    Filters      = Fxied_filters(FILE_NAME_PATH, fs)

    N  = fs + 1023 
    noise_1 = BandlimitedNoise_generation_tensor(7000, 210, fs, N, 100)
    # noise_2 = BandlimitedNoise_generation_tensor(4600, 500, fs, N, 90)

    # plot_specgram(noise_1, fs, title="Spectrogram")
    # plot_specgram(noise_2, fs, title="Spectrogram")

    MODEL_PATH = "feedforwardnet.pth"
    Predictor  = OneD_CNN_Predictor(MODEL_PATH)

    # out2       = Predictor.cosSimilarity_minmax(noise_1, noise_2)
    # print(f"The cos similarity is {out2:0.4f}.")
    # F_start = np.linspace(100, 5000, 300)

    # no_pre = []
    # for f_star in F_start:
    #     noise_2 = BandlimitedNoise_generation_tensor(f_star, 1000, fs, N, 90)
    #     out2    = Predictor.cosSimilarity_minmax(noise_1, noise_2)
    #     no_pre.append(out2)
    
    # plt.plot(F_start, no_pre)
    # plt.grid()
    # plt.show()

    noise_2 = Filters.Charactors[3]
    noise_3 = Filters.Charactors[4]
    plot_specgram(noise_1,fs)
    plot_specgram(noise_2,fs)
    plot_specgram(noise_3,fs)
    out2    = Predictor.cosSimilarity_minmax(noise_1,noise_2)
    out3    = Predictor.cosSimilarity_minmax(noise_1,noise_3)
    print(out2)
    print(out3)

    similarity_rato = []
    for ii in range(Filters.len):
        similarity_rato.append(Predictor.cosSimilarity_minmax(noise_1, Filters.Charactors[ii]))
    
    index = np.argmax(similarity_rato)
    print(f'The selected control filter is C{index}')

    #----Class tsting-----%
    CNN_classfier = Filter_ID_predictor(MODEL_PATH, FILE_NAME_PATH, fs)

    print(CNN_classfier.predic_ID(noise_1))
    i = 0 
