import os 
from DataSet_construction_v2 import DatasetSheet
import scipy.io as sio

import matplotlib.pyplot as plt
from Fixed_Filter_noise_cancellation import Fxied_filters
from Disturbance_generation import Disturbance_generation_from_real_noise
from Noise_reduction_level_calculat import NR_level_compute

import torchaudio
import torch
from scipy import signal
import  numpy as np 
import math

import progressbar
#----------------------------------------------------
def additional_noise(signal, snr_db):
    signal_power     = signal.norm(p=2)
    length           = signal.shape[1]
    additional_noise = np.random.randn(length)
    additional_noise = torch.from_numpy(additional_noise).type(torch.float32).unsqueeze(0)
    noise_power      = additional_noise.norm(p=2)
    snr              = math.exp(snr_db / 10)
    scale            = snr * noise_power / signal_power
    noisy_signal     = signal + additional_noise/scale
    return noisy_signal

#----------------------------------------------------
def plot_frequency(Wc_matrix):
    fs=16000
    f, Pper_spec = signal.periodogram(Wc_matrix, fs, 'flattop', scaling='spectrum')
    plt.semilogy(f, Pper_spec)
    plt.xlabel('frequency [Hz]')
    plt.ylabel('PSD')
    plt.grid()
    plt.show()

#----------------------------------------------------
# Function loading_paths_from_MAT（）
#----------------------------------------------------
def loading_paths_from_MAT(folder = 'Duct_path'
                           ,Pri_path_file_name = 'Primary_path.mat'
                           ,Sec_path_file_name ='Secondary_path.mat'):
    Primay_path_file, Secondary_path_file = os.path.join(folder, Pri_path_file_name), os.path.join(folder,Sec_path_file_name)
    Pri_dfs, Secon_dfs = sio.loadmat(Primay_path_file), sio.loadmat(Secondary_path_file)
    Pri_path, Secon_path = Pri_dfs['Pri_path'].squeeze(), Secon_dfs['Sec_path'].squeeze()
    return Pri_path, Secon_path
#----------------------------------------------------
# Function: read_all_file_from_folder()
# Description : Get the file with desire sufix. 
#----------------------------------------------------
def read_all_file_from_folder(folder_path, file_sufix):
    f_names = []
    for (dirpath, dirnames, filenames) in os.walk(folder_path):
        for file_name in filenames:
            root_ext = os.path.splitext(file_name)
            if root_ext[1] == file_sufix:
                f_names.append(file_name)
    return f_names

#-----------------------------------------------------
# Class: automatic_lable_machine
#-----------------------------------------------------
class Automatic_label_machine():
    
    def __init__(self, folder_path, path_mat, Index_file_name='auto_index.csv', **kwargs):
        '''
        Parameters:
            param1 - folder_path: the dirctory of the trainning set 
            param2 - path_mat: the dirctory of the pre-trained control filter file 
            param3 - Index_file_name: the output file name 
        '''
        sufix = '.wav'
        fs    = 16000 
        self.fs       = fs
        self.folder   = folder_path
        self.f_names  = read_all_file_from_folder(folder_path=folder_path, file_sufix=sufix)
        self.Fixed_filters = Fxied_filters(MATFILE_PATH=path_mat,fs=fs) 
        self.lable_inex = DatasetSheet(folder=folder_path, filename=Index_file_name)
        self.Pri_path, self.Secon_path = loading_paths_from_MAT(folder= kwargs['folder']
                                                                ,Pri_path_file_name= kwargs['Pri_path_file_name']
                                                                ,Sec_path_file_name= kwargs['Sec_path_file_name'])
    
    def label_out(self, SNR):
        '''
        Parameters:
            param1 - SNR: signal to noise ratio 
        '''
        bar = progressbar.ProgressBar(maxval=len(self.f_names), \
        widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        
        print('=====================Start labeling ============================')
        print(f'The total .wav file numer is {len(self.f_names)}')
        bar.start()
        for iterate, file_wave in enumerate(self.f_names):
            filePath  = os.path.join(self.folder, file_wave)
            signal,_  = torchaudio.load(filePath)

            Dir, Fx, _= Disturbance_generation_from_real_noise(fs=self.fs, Repet= 3, wave_from=signal,Pri_path=self.Pri_path,Sec_path=self.Secon_path)
            
            Fx        = additional_noise(signal=Fx.unsqueeze(0), snr_db=SNR)
            Fx        = Fx[0,:]
            Nr_levels = []
            for ii in range(self.Fixed_filters.len):
                Err = self.Fixed_filters.cancellator(classID=ii, Fx=Fx, Dir=Dir)
                Nr_levels.append(NR_level_compute(Disturbance=Dir.numpy(),Error=Err.numpy()))
            
            label = Nr_levels.index(max(Nr_levels))
            self.lable_inex.add_data_to_file(wave_file=file_wave,class_ID=label)
            bar.update(iterate)
        bar.finish()
        self.lable_inex.flush()
        print('=======================End labeling ============================')
        
        
#-----------------------------------------------------
if __name__=="__main__":
    # folder_path = 'Duct_path'
    # f_names = read_all_file_from_folder(folder_path=folder_path, file_sufix='.csv')
    
    # print(f_names)
    # Pri_path, Secon_path = loading_paths_from_MAT()
    # # Drawing the impulse response of the primary path
    # plt.title('The response of the primary path')
    # plt.plot(Secon_path)
    # plt.ylabel('Amplitude')
    # plt.xlabel('Time')
    # plt.grid()
    # plt.show()
    
    # plot_frequency(Secon_path)
    
    # Pre-trained control filter file 
    control_filter_mat_file  = 'Control_filter_from_15frequencies.mat'
    # Training set and label out file 
    training_set_folder_path = 'Testing_data'
    new_training_index       = 'auto_index.csv'
    # Measured path file 
    folder                   = 'Duct_path'
    Pri_path_file_name       = 'Primary_path.mat'
    Sec_path_file_name       = 'Secondary_path.mat'
    Labeler = Automatic_label_machine(folder_path          = training_set_folder_path
                                      , path_mat           = control_filter_mat_file
                                      , Index_file_name    = new_training_index
                                      , folder             = folder
                                      , Pri_path_file_name = Pri_path_file_name
                                      , Sec_path_file_name = Sec_path_file_name)
    Labeler.label_out(SNR = 10)
    