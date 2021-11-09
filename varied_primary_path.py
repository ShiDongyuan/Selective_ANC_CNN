import numpy as np 
import math
import torch
from scipy import signal, misc

from Reading_path_tst import loading_paths_from_MAT
import matplotlib.pyplot as plt 


#---------------------------------------------------------
# Function    : additional_noise_to_wav()
# Description : Adding the additional noise into the data
#---------------------------------------------------------
def additional_noise_to_wav(snr_db=None, speech=None):
    if snr_db == None:
        noisy_speech = speech 
    else:
        N     = len(speech)
        noise = np.random.randn(N)
        snr   = math.exp(snr_db / 10)
        speech_power = np.var(speech)
        noise_power  = np.var(noise)
        scale = np.sqrt(snr * noise_power / speech_power)
        noisy_speech = speech + noise/scale
    return noisy_speech

def varid_primary_path_distrubance(fs, T, f_vector, Pri_path, Sec_path,SNR):
    t     = np.arange(0,T,1/fs).reshape(-1,1)
    len_f = 1024
    b2    = signal.firwin(len_f, f_vector, pass_zero='bandpass', window ='hamming',fs=fs)
    xin   = np.random.randn(len(t))
    Re    = signal.lfilter(b2,1,xin)
    
    Path_v = np.zeros((len(SNR)+1,len(Pri_path)))

    for ii, snr_db in enumerate(SNR):
        if ii == 0 :
            pass
        else:
            Pri_path     = additional_noise_to_wav(snr_db,Pri_path)
        Path_v[ii,:] = Pri_path
        disturbance  = signal.lfilter(Pri_path,1,Re)
        disturbance  = disturbance[fs:]
        fx           = signal.lfilter(Sec_path,1,Re)
        fx           = fx[fs:]
        
        if ii == 0:
            Dis = disturbance
            Fx  = fx
            Pri_noise    = Re[fs:]
        else:
            Dis = np.concatenate((Dis,disturbance),axis=0)
            Fx  = np.concatenate((Fx,fx),axis=0)
            Pri_noise = np.concatenate((Pri_noise,Pri_noise),axis=0)
    return torch.from_numpy(Dis).type(torch.float), torch.from_numpy(Fx).type(torch.float), torch.from_numpy(Pri_noise).type(torch.float), Path_v

if __name__ == "__main__":
    Pri_path, Secon_path = loading_paths_from_MAT()
    Dis, Fx, Path_v = varid_primary_path_distrubance(fs=16000, T=9, f_vector=[600, 1200], Pri_path=Pri_path, Sec_path= Secon_path, SNR =[120, 30, 10])
    plt.title('The response of the primary path')
    plt.plot(Path_v[0,:])
    plt.ylabel('Amplitude')
    plt.xlabel('Time')
    plt.grid()
    plt.show()