import numpy as np
import torch 
import torch.nn as nn
import torch.optim as optim

import scipy.io as sio
import scipy.signal as signal

# from Tst_CNN_predictor import plot_specgram
# from Tst_CNN_predictor import plot_waveform

import scipy.signal as signal
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq, ifft
#-----------------------------------------------------------------------------------
# Class type  : Fxied_filters 
# Description : Loading pre-trained filter from .mat file
#-----------------------------------------------------------------------------------
class Fxied_filters():

    def __init__(self, MATFILE_PATH, fs):
        mat_contents    = sio.loadmat(MATFILE_PATH)
        self.Wc_vectors = mat_contents['Wc_v']
        self.len        = len(self.Wc_vectors)
        self.filterlen  = self.Wc_vectors.shape[1]
        self.Charactors = torch.zeros([self.len, 1, fs], dtype=torch.float)
        self.fs         = fs 

        for ii in range(self.len):
            self.Charactors[ii] = self.frequency_charactors_tensor(ii)
    
    def cancellator(self, classID, Fx, Dir):
        Yt = signal.lfilter(self.Wc_vectors[classID,:], 1, Fx)
        Er = Dir - Yt 
        return Er 

    def frequency_charactors_tensor(self, classID):
        fs    = self.fs
        N     = fs + self.filterlen 
        xin   = np.random.randn(N)
        #print(self.Wc_vectors[classID,:])
        yout  = signal.lfilter(self.Wc_vectors[classID,:],1,xin)
        yout  = yout[self.filterlen:]
        # Standarlize 
        yout = yout/np.sqrt(np.var(yout))
        # return a tensor of [1 x sample rate]
        return torch.from_numpy(yout).type(torch.float).unsqueeze(0)
    

#<<<<<---------------------------------Main function-------------------------------------------->>>
if __name__ == "__main__":
    fs           = 16000
    MATFILE_PATH = 'Pre-train Control filter.mat'
    Filters      = Fxied_filters(MATFILE_PATH, fs)
    
    Fre_feature  = Filters.frequency_charactors_tensor(9)
    plot_specgram(Fre_feature, fs)
    plot_waveform(Fre_feature, fs)
    print(Filters.Charactors[2].shape)

    for ii in range(Filters.len):
        Re = Filters.Charactors[ii,0].numpy()

        fig, ax0= plt.subplots()

        N  = len(Re)
        yf = fft(Re)
        xf = fftfreq(N, 1/16000)[:N//2]
        ax0.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
        ax0.grid()

        fig.savefig(f"Figure\\Freuency_of_control_filter_C{ii}.jpg",dpi=600)

    i = 0 

        