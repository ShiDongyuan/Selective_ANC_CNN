#    ____  ____  ____    _          _
#   |  _ \/ ___||  _ \  | |    __ _| |__
#   | | | \___ \| |_) | | |   / _` | '_ \
#   | |_| |___) |  __/  | |__| (_| | |_) |
#   |____/|____/|_|     |_____\__,_|_.__/
#
import numpy as np
import os

import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
import scipy.signal as signal
import math
import pandas as pd 
from   Bcolors import bcolors

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

def BandlimitedNoise_generation(f_star, Bandwidth, fs, N):
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
    #----------------------------------------------------
    return Noise/np.sqrt(np.var(Noise))

def additional_noise(signal, snr_db):
    signal_power     = signal.norm(p=2)
    length           = signal.shape[1]
    additional_noise = np.random.randn(length)
    additional_noise = torch.from_numpy(additional_noise).type(torch.float32).unsqueeze(0)
    noise_power      = additional_noise.norm(p=2)
    snr              = math.exp(snr_db / 10)
    scale            = snr * noise_power / signal_power
    noisy_signal     = (scale * signal + additional_noise) / 2
    return noisy_signal
#-------------------------------------------------------------
# Class : SoundGnereator 
#-------------------------------------------------------------
class SoundGenerator:
    def __init__(self, fs, folder):
        self.fs     = fs 
        self.len    = fs + 1023 
        self.folder = folder 
        self.Num    = 0 
        try: 
            os.mkdir(folder)
        except:
            print("folder exists")
    
    def _construct_(self):
        self.Num  = self.Num + 1 
        f_star    = np.random.uniform(20, 7880, 1)
        bandWidth = np.random.uniform(1,7880-f_star,1)
        f_end     = f_star + bandWidth
        filename  = f'{self.Num}_Frequency_from_'+ f'{f_star[0]:.0f}_to_{f_end[0]:.0f}_Hz.wav'
        filePath  = os.path.join(self.folder, filename)
        noise     = BandlimitedNoise_generation(f_star[0], bandWidth[0], self.fs, self.len)
        noise     = torch.from_numpy(noise).type(torch.float32).unsqueeze(0)
        torchaudio.save(filePath, noise, self.fs)
        return f_star[0], f_end[0], filename
    
    def _construct_A(self):
        self.Num  = self.Num + 1 
        f_star    = np.random.uniform(20, 7880, 1)
        bandWidth = np.random.uniform(1,7880-f_star,1)
        f_end     = f_star + bandWidth
        filename  = f'{self.Num}_Frequency_from_'+ f'{f_star[0]:.0f}_to_{f_end[0]:.0f}_Hz_A.wav'
        filePath  = os.path.join(self.folder, filename)
        noise     = BandlimitedNoise_generation(f_star[0], bandWidth[0], self.fs, self.len)
        noise     = torch.from_numpy(noise).type(torch.float32).unsqueeze(0)
        snr_db    = np.random.uniform(3, 60, 1)
        noise     = additional_noise(noise, snr_db)
        torchaudio.save(filePath, noise, self.fs)
        return f_star[0], f_end[0], filename
    
    def _balance_construct(self, Fre_noise_vector):
        self.Num  = self.Num + 1 
        filename  = f'{self.Num}_Frequency_from_'+ f'{Fre_noise_vector[0]:.0f}_to_{Fre_noise_vector[1]:.0f}_Hz.wav'
        filePath  = os.path.join(self.folder, filename)
        noise     = BandlimitedNoise_generation(Fre_noise_vector[0], Fre_noise_vector[1]-Fre_noise_vector[0], self.fs, self.len)
        noise     = torch.from_numpy(noise).type(torch.float32).unsqueeze(0)
        torchaudio.save(filePath, noise, self.fs)
        return filename
    
    def _balance_construct_A(self, Fre_noise_vector):
        self.Num  = self.Num + 1 
        filename  = f'{self.Num}_Frequency_from_'+ f'{Fre_noise_vector[0]:.0f}_to_{Fre_noise_vector[1]:.0f}_Hz.wav'
        filePath  = os.path.join(self.folder, filename)
        noise     = BandlimitedNoise_generation(Fre_noise_vector[0], Fre_noise_vector[1]-Fre_noise_vector[0], self.fs, self.len)
        noise     = torch.from_numpy(noise).type(torch.float32).unsqueeze(0)
        snr_db    = np.random.uniform(3, 60, 1)
        noise     = additional_noise(noise, snr_db)
        torchaudio.save(filePath, noise, self.fs)
        return filename

class DatasetSheet:
    
    def __init__(self, folder, filename):
        self.filename = filename 
        self.folder   = folder
        try: 
            os.mkdir(folder, 755)
        except:
            print("folder exists")
        self.path     = os.path.join(folder, filename)
    
    def add_data_to_file(self, wave_file, class_ID):
        dict         = {'File_path': [wave_file], 'Class_ID': [class_ID]}
        df           = pd.DataFrame(dict)
        
        with open(self.path, mode = 'a') as f:
            df.to_csv(f, header=f.tell()==0)
        
    def flush(self):
        dc       = pd.read_csv(self.path, index_col=0)
        dc.index = range(len(dc))
        dc.to_csv(self.path)

#-------------------------------------------------------------
# Function    : SimilarityRato(f1_min, f1_max, f2_min, f2_max)
# Description : Geting Class ID of frequency band.  
#-------------------------------------------------------------
def SimilarityRato(f1_min, f1_max, f2_min, f2_max):
    if (f1_min <= f2_min):
        if (f1_max <= f2_min):
            return 0
        elif (f2_min <= f1_max) & (f1_max <= f2_max):
            return (f1_max-f2_min)/(f2_max-f1_min)
        else:
            return (f2_max-f2_min)/(f1_max-f1_min)
    else:
        if (f2_max <= f1_min):
            return 0
        elif (f1_min <= f2_max)&(f2_max <= f1_max):
            return (f2_max-f1_min)/(f1_max-f2_min)
        else:
            return (f1_max-f1_min)/(f2_max-f2_min)
#-----------------------------------------------------------------
# Class     :   ClassID_Calculator
#-----------------------------------------------------------------
class ClassID_Calculator:
    
    def __init__(self, levels):
        self.f_vector = levels
        self.len = len(self.f_vector)
            
    def _get_ID_(self, f_low, f_high):
        SimlarityRatio = []
        for ii in range(self.len):
            SimlarityRatio.append(SimilarityRato(f_low, f_high, self.f_vector[ii][0],self.f_vector[ii][1]))
        ID = SimlarityRatio.index(max(SimlarityRatio))
        return ID, SimlarityRatio

#------------------------------------------------------------------------------------------
# Function     :   Generating Dataset as given frequency band (It comes from main function) 
#-------------------------------------------------------------------------------------------
def Generating_Dataset_as_Given_Frequencybands(N_sample_each_class, F_bands, Folder_name):
    import progressbar
    
    file_name   = "Index.csv"
    
    datasheet     = DatasetSheet(Folder_name,file_name)
    Generator     = SoundGenerator(fs=16000, folder = Folder_name)
    
    Fre_noise_band, Fre_target     = Generating_balance_sampleset_frequency_band_vector(Frequecy_band=F_bands, Sample_set_number=N_sample_each_class)
    Fre_noise_band_A, Fre_target_A = Generating_balance_sampleset_frequency_band_vector(Frequecy_band=F_bands, Sample_set_number=N_sample_each_class)
    print(bcolors.RED + f'Each sample set has {len(Fre_target)+len(Fre_target_A)} !!!' + bcolors.ENDC)
    
    bar = progressbar.ProgressBar(maxval=len(Fre_target), \
        widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    ii = 0 
    bar.start()
    for Fre_noise_vector, Fre_target_element, Fre_noise_vector_A, Fre_target_element_A  in zip(Fre_noise_band, Fre_target, Fre_noise_band_A, Fre_target_A):
        filePath = Generator._balance_construct(Fre_noise_vector=Fre_noise_vector)
        datasheet.add_data_to_file(filePath,Fre_target_element)
        
        filePath = Generator._balance_construct_A(Fre_noise_vector=Fre_noise_vector_A)
        datasheet.add_data_to_file(filePath,Fre_target_element_A)
        ii +=1
        bar.update(ii)
    datasheet.flush()
    bar.finish()
#------------------------------------------------------------------------------------------
# Function     :   Generating_balance_sampleset_frequency_band_vector
# Description  :   Creating the balance number of refeqncy band 
#------------------------------------------------------------------------------------------
def Generating_balance_sampleset_frequency_band_vector(Frequecy_band, Sample_set_number):
    Max_number     = Sample_set_number 
    ID_calculator  = ClassID_Calculator(Frequecy_band)
    Class_count    = np.zeros(len(Frequecy_band))
    Class_num      = len(Frequecy_band)
    Fre_noise_band = []
    Fre_target     = []
    #-------------------------------------------------------------------------------------
    Continue_flag  = True
    while Continue_flag:
        F_band = np.sort(np.random.uniform(20, 7880, 2))
        if F_band[0] == F_band[1]:
            continue
        ID,_     = ID_calculator._get_ID_(f_low=F_band[0], f_high=F_band[1])
        if Class_count[ID] < Max_number:
            Fre_noise_band.append(F_band)
            Fre_target.append(ID)
            Class_count[ID] +=1 
        
        if np.sum(Class_count == Max_number)==Class_num:
            Continue_flag = False
        else:
            Continue_flag = True 
    print(bcolors.OKGREEN + f' Have created {Class_num} balance frequency band for datast !!!' + bcolors.ENDC)
    return Fre_noise_band, Fre_target
    


if __name__=='__main__':
    generate_set_for_class = 15
    if generate_set_for_class == 4:
        F_band = np.sort(np.random.uniform(20, 7880, 2))
        print(F_band.shape)
        Frequecy_band = [[20, 550], [450, 1200], [1000, 2700],[2500, 4500],[4400, 7980]]
        print(len(Frequecy_band))
        
        
        Fre_noise_band, Fre_target = Generating_balance_sampleset_frequency_band_vector(Frequecy_band=Frequecy_band, Sample_set_number=10)
        print(Fre_noise_band[0][0])
        print(Fre_target)
        
        File_name_of_dataset_list = ['Train_dataset_of_5frequencybands'
                                    , 'Validate_dataset_of_5frequencybands'
                                    , 'Testing_dataset_of_5frequencybands']
        
        for folder_name in File_name_of_dataset_list:
            Generating_Dataset_as_Given_Frequencybands(N_sample_each_class=200, F_bands=Frequecy_band,Folder_name=folder_name)
            print(f'Has finihsed {folder_name} !!!!')
    else: 
        level = 4 
        fs    = 16000  
        F_band = []   
        for level in range(level):
                a_vector,_    = frequencyband_design(level,fs) 
                F_band       += a_vector
        
        Folder_name_list_of_data_set =['Training_data', 'Validate_data', 'Testing_data']
        print(40000//len(F_band))
        N_sample_list =[(40000//len(F_band)), (1000//len(F_band)), (1000//len(F_band))]
        
        for folder_name, N_sample_element in zip(Folder_name_list_of_data_set, N_sample_list):
            Generating_Dataset_as_Given_Frequencybands(N_sample_each_class=N_sample_element, F_bands=F_band,Folder_name=folder_name)
            print(bcolors.OKCYAN + f'Has finihsed {folder_name} !!!!' + bcolors.ENDC)
    
   