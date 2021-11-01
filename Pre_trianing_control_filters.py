from Reading_path_tst import loading_paths
from Disturbance_generation import DistDisturbance_reference_generation_from_Fvector
import numpy as np
from FxLMS_algorithm import FxLMS_algroithm, train_fxlms_algorithm
import matplotlib.pyplot as plt

from scipy.io import savemat

Frequecy_band = [[20, 550], [450, 1200], [1000, 2700],[2500, 4500],[4400, 7980]]

def save_mat__(FILE_NAME_PATH, Wc):
    mdict= {'Wc_v': Wc}
    savemat(FILE_NAME_PATH, mdict)

def main():
    FILE_NAME_PATH = 'Control_filter_from_5frequencies.mat'
    # Configurating the system parameters
    fs = 16000 
    T  = 30 
    Len_control = 1024 
    
    # Loading the primary and secondary path
    Pri_path, Secon_path = loading_paths() 
    
    # Training the control filters from the defined frequency band 
    num_filters = len(Frequecy_band)
    Wc_matrix   = np.zeros((num_filters, Len_control), dtype=float)
    
    for ii, F_vector in enumerate( Frequecy_band):
        Dis, Fx = DistDisturbance_reference_generation_from_Fvector(fs=fs, T= T, f_vector=F_vector, Pri_path=Pri_path, Sec_path=Secon_path)
        controller = FxLMS_algroithm(Len=Len_control)
        Wc_matrix[ii] = controller._get_coeff_()
        Erro = train_fxlms_algorithm(Model=controller,Ref=Fx, Disturbance=Dis)
        
        # Drawing the impulse response of the primary path
        plt.title('The error signal of the FxLMS algorithm')
        plt.plot(Erro)
        plt.ylabel('Amplitude')
        plt.xlabel('Time')
        plt.grid()
        plt.show()
        
        save_mat__(FILE_NAME_PATH, Wc_matrix)

if __name__ == "__main__":
    main()