import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq, ifft

import torch 
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from scipy.io import savemat

import pandas as pd
from scipy.io import savemat
#-----------------------------------------------------------------------------------
# Class       : frequencyband_design()
# Description : The function is utilized to devide the full frequency band into 
# several equal frequency components.
#-----------------------------------------------------------------------------------
def frequencyband_design(level,fs):
    # the number of filter equals 2^level.
    # fs represents the sampling rate. 
    Num = 2**level
    # Computing the start and end of the frequency band.
    #----------------------------------------------------
    F_vector = []
    f_start  = 20
    f_marge  = 20 
    # the wideth of thefrequency band
    width    = (fs/2-f_start-f_marge)//Num 
    #----------------------------------------------------
    for ii in range(Num):
        f_end   = f_start + width 
        F_vector.append([f_start,f_end])
        f_start = f_end 
    #----------------------------------------------------
    return F_vector, width

#-----------------------------------------------------------------------------------
# Class type  : Filter design
# Description : Design filter group by the configure vector 
#-----------------------------------------------------------------------------------
class Filter_designer():
    
    def __init__(self, filter_len, F_vector, fs):

        self.filter_len = filter_len
        self.filter_num = len(F_vector)
        self.wc         = np.zeros((self.filter_num, self.filter_len))
        for i in range(self.filter_num):
            self.wc[i,:] = signal.firwin(self.filter_len, F_vector[i], pass_zero='bandpass', window ='hamming',fs=fs) 
        
    
    def __save_mat__(self, FILE_NAME_PATH):
        mdict= {'Wc_v': self.wc}
        savemat(FILE_NAME_PATH, mdict)

#-----------------------------------------------------------------------------------
#------------------->> Main() <<-----------------------
#-----------------------------------------------------------------------------------
if __name__ == "__main__":
    FILE_NAME_PATH = "Bandlimited_filter.mat"
    fs             = 16000 
    level          = 4 #4 

    F_vector = []
    for i in range(level):
        F_vec, _    = frequencyband_design(i, fs)
        F_vector   += F_vec

    Filters = Filter_designer(filter_len=1024, F_vector= F_vector, fs=fs)
    Filters.__save_mat__(FILE_NAME_PATH)
    print(Filters.filter_num)
#--------------------------------------------------------------------
# Design own bandfilter 
    from DataSet_construction_DesignBand import F_LEVEL
    FILE_NAME_PATH1 = "DesignBand_filter_v1.mat"
    Filters = Filter_designer(filter_len=1024, F_vector= F_LEVEL, fs=fs)
    Filters.__save_mat__(FILE_NAME_PATH1)
    print(Filters.filter_num)

    i = 0 

    