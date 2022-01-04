from Automatic_label_machine import loading_paths_from_MAT
from Disturbance_generation import DistDisturbance_reference_generation_from_Fvector
import numpy as np
from FxLMS_algorithm import FxLMS_algroithm, train_fxlms_algorithm
import matplotlib.pyplot as plt
from Filter_design import frequencyband_design

from scipy.io import savemat
from scipy import signal, misc

def save_mat__(FILE_NAME_PATH, Wc):
    mdict= {'Wc_v': Wc}
    savemat(FILE_NAME_PATH, mdict)

def main():
    FILE_NAME_PATH = 'Control_filter_from_15frequencies.mat'
    # Configurating the system parameters
    fs = 16000 
    T  = 30 
    Len_control = 1024 
    level          = 4 #4 
    
    Frequecy_band = []
    for i in range(level):
        F_vec, _       = frequencyband_design(i, fs)
        Frequecy_band += F_vec
    
    # Loading the primary and secondary path
    # Pri_path, Secon_path = loading_paths() 
    Pri_path, Secon_path = loading_paths_from_MAT()
    
    # Training the control filters from the defined frequency band 
    num_filters = len(Frequecy_band)
    Wc_matrix   = np.zeros((num_filters, Len_control), dtype=float)
    
    print(Frequecy_band)
    
    
    for ii, F_vector in enumerate( Frequecy_band):
        print(F_vector)
        Dis, Fx = DistDisturbance_reference_generation_from_Fvector(fs=fs, T= T, f_vector=F_vector, Pri_path=Pri_path, Sec_path=Secon_path)
        controller = FxLMS_algroithm(Len=Len_control)
        
        Erro = train_fxlms_algorithm(Model=controller,Ref=Fx, Disturbance=Dis)
        Wc_matrix[ii] = controller._get_coeff_()
        
        # Drawing the impulse response of the primary path
        plt.title('The error signal of the FxLMS algorithm')
        plt.plot(Erro)
        plt.ylabel('Amplitude')
        plt.xlabel('Time')
        plt.grid()
        plt.show()
        
        fs=16000
        f, Pper_spec = signal.periodogram(Wc_matrix[ii] , fs, 'flattop', scaling='spectrum')
        plt.semilogy(f, Pper_spec)
        plt.xlabel('frequency [Hz]')
        plt.ylabel('PSD')
        plt.grid()
        plt.show()
        
    save_mat__(FILE_NAME_PATH, Wc_matrix)

if __name__ == "__main__":
    main()