import torch
import os
import numpy as np
from torch import nn

from Tst_CNN_predicotr_v1 import Filter_ID_predictor
from Fixed_Filter_noise_cancellation import Fxied_filters
from ONED_CNN_PRE import OneD_CNN_Predictor
import scipy.signal as signal

from Filter_design import Boardband_Filter_Desgin_as_Given_Freqeuencybands

#-------------------------------------------------------------
# Function  :   load_weigth_for_model()
# Loading the weights to model from pre-trained coefficients 
#-------------------------------------------------------------
def load_weigth_for_model(model, pretrained_path, device):
    model_dict      = model.state_dict()
    pretrained_dict = torch.load(pretrained_path,map_location= device)
    #=========================================================
    for k, v in model_dict.items():
        model_dict[k] = pretrained_dict[k]
    #=========================================================
    model.load_state_dict(model_dict)
#-------------------------------------------------------------
# Function: Is multiple length of samples
#-------------------------------------------------------------
def Casting_multiple_time_length_of_primary_noise(primary_noise,fs):
    assert  primary_noise.shape[0] == 1, 'The dimension of the primary noise should be [1 x samples] !!!'
    cast_len = primary_noise.shape[1] - primary_noise.shape[1]%fs
    return primary_noise[:,:cast_len]

def Casting_single_time_length_of_training_noise(filter_training_noise,fs):
    assert filter_training_noise.dim() == 3, 'The dimension of the training noise should be 3 !!!'
    print(filter_training_noise[:,:,:fs].shape)
    return filter_training_noise[:,:,:fs]

#------------------------------------------------------------
# Function : Generating the testing bordband noise 
#------------------------------------------------------------
def Generating_boardband_noise_wavefrom_tensor(Wc_F, Seconds, fs):
    filter_len = 1024 
    bandpass_filter = signal.firwin(filter_len, Wc_F, pass_zero='bandpass', window ='hamming',fs=fs) 
    N = filter_len + Seconds*fs
    xin = np.random.randn(N)
    y   = signal.lfilter(bandpass_filter,1,xin)
    yout= y[filter_len:]
    # Standarlize 
    yout = yout/np.sqrt(np.var(yout))
    # return a tensor of [1 x sample rate]
    return torch.from_numpy(yout).type(torch.float).unsqueeze(0)
    
#-------------------------------------------------------------
# Class :   Control_filter_Index_predictor
#-------------------------------------------------------------
class Control_filter_Index_predictor(OneD_CNN_Predictor):
    
    def __init__(self,MODEL_PATH,device,filter_training_noise,fs):
        
        OneD_CNN_Predictor.__init__(self,MODEL_PATH,device)
        # Checking the length of the training noise 
        assert filter_training_noise.dim() == 3, 'The dimension of the training noise should be 3 !!!'
        assert filter_training_noise.shape[2]%fs == 0, 'The length of the training noise sample should be 1 second!'
        # Detach the information of the the training nosie 
        self.frequency_charactors_tensor = filter_training_noise 
        self.len_of_class                = filter_training_noise.shape[0]
        self.fs                          = fs 
    
    def predic_ID(self, noise):
        similarity_rato = []
        for ii in range(self.len_of_class):
            similarity_rato.append(self.cosSimilarity_minmax(noise, self.frequency_charactors_tensor[ii]))
        index = np.argmax(similarity_rato)
        return index
    
    def predic_ID_vector(self, primary_noise):
        # Checking the length of the primary noise.
        assert  primary_noise.shape[0] == 1, 'The dimension of the primary noise should be [1 x samples] !!!'
        assert  primary_noise.shape[1] % self.fs == 0, 'The length of the primary noise is not an integral multiple of fs.'
        # Computing how many seconds the primary noise containt.
        Time_len              = int(primary_noise.shape[1]/self.fs) 
        print(Time_len)
        print(f'The primary nosie has {Time_len} seconds !!!')
        # Bulding the matric of the primary noise [times x 1 x fs ]
        primary_noise_vectors = primary_noise.reshape(Time_len,self.fs).unsqueeze(1)
        
        
        # Implementing the noise classification for each frame whose length is 1 second. 
        ID_vector = []
        for ii in range(Time_len):
            ID_vector.append(self.predic_ID(primary_noise_vectors[ii]))
        
        return ID_vector
#-------------------------------------------------------------    
# function: main()
# This function is used to test and debug the code 
#-------------------------------------------------------------
def main():
    Frequecy_band = [[20, 550], [450, 1200], [1000, 2700],[2500, 4500],[4400, 7980]]
    
    # Creating the pre-trained band filter for 5 different frequency band    
    Filter_mat_name = 'Boardband_filter_from_5frequencybands.mat'
    if not os.path.exists(Filter_mat_name):
        Boardband_Filter_Desgin_as_Given_Freqeuencybands(MAT_filename=Filter_mat_name, F_bands=Frequecy_band,fs=16000)
    else:
        print("Data of " + Filter_mat_name + ' is existed !!!')  
    
    Fxied_control_filter = Fxied_filters(MATFILE_PATH=Filter_mat_name, fs=16000)
    
    Charactors=Casting_single_time_length_of_training_noise(Fxied_control_filter.Charactors,fs=16000)
    
    # cnn modle path     
    MODEL_PTH = "feedforwardnet_Nway_v2.pth"#'feedforwardnet_LMSoftmax_v4.pth'#'feedforwardnet_v1.pth'
    device    = "cpu"
    
    Pre_trained_control_filter_ID_pridector = Control_filter_Index_predictor(MODEL_PATH=MODEL_PTH
                                                                             ,device= device
                                                                             ,filter_training_noise=Charactors
                                                                             ,fs=16000)
    
    primary_noise = Generating_boardband_noise_wavefrom_tensor([1800, 2010],6,fs=16000)
    print(primary_noise.shape)
    Id_vector = Pre_trained_control_filter_ID_pridector.predic_ID_vector(primary_noise)
    print(Id_vector)
    pass
    
if __name__ == "__main__":
    main()
    
        