import os 
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt 

from scipy import signal, misc
import scipy.io as sio


from Disturbance_generation import DistDisturbance_reference_generation_from_Fvector


def loading_paths(folder="Duct_path", Pri_path_file_name = "Primary Path.csv", Sec_path_file_name="Secondary Path.csv"):
    Primay_path_file, Secondary_path_file = os.path.join(folder,Pri_path_file_name), os.path.join(folder,Sec_path_file_name)
    Pri_dfs, Secon_dfs   = pd.read_csv(Primay_path_file), pd.read_csv(Secondary_path_file)
    Pri_path, Secon_path = np.array(Pri_dfs['Amplitude - Plot 0']), np.array(Secon_dfs['Amplitude - Plot 0'])
    return Pri_path, Secon_path

def loading_paths_from_MAT(folder = 'Pz and Sz'
                           ,subfolder = 'measured'
                           ,Pri_path_file_name = 'Pz1.mat'
                           ,Sec_path_file_name ='Sz.mat'):
    Primay_path_file, Secondary_path_file = os.path.join(folder, subfolder, Pri_path_file_name), os.path.join(folder,subfolder, Sec_path_file_name)
    Pri_dfs, Secon_dfs = sio.loadmat(Primay_path_file), sio.loadmat(Secondary_path_file)
    Pri_path, Secon_path = Pri_dfs['PZ1'].squeeze(), Secon_dfs['S'].squeeze()
    return Pri_path, Secon_path


if __name__=="__main__":
    
    Pri_path, Secon_path = loading_paths()
    
    plt.title('The response of the primary path')
    plt.plot(Pri_path)
    plt.ylabel('Amplitude')
    plt.xlabel('Time')
    plt.grid()
    plt.show()
    
    # plt.title('The response of the primary path')
    # plt.plot(Secon_path)
    # plt.ylabel('Amplitude')
    # plt.xlabel('Time')
    # plt.grid()
    # plt.show()
    
    # Dis, Fx = DistDisturbance_reference_generation_from_Fvector(fs = 16000, T= 5, f_vector=[500, 1500], Pri_path=Pri_path, Sec_path=Secon_path )
    
    # plt.title('The response of the primary path')
    # plt.plot(Dis)
    # plt.ylabel('Amplitude')
    # plt.xlabel('Time')
    # plt.grid()
    # plt.show()
    
    # # Drawing the frequency spectrum of the disturbance 
    # f, Pper_spec = signal.periodogram(Dis, 16000, 'flattop', scaling='spectrum')
    # plt.semilogy(f, Pper_spec)
    # plt.xlabel('frequency [Hz]')
    # plt.ylabel('PSD')
    # plt.grid()
    # plt.show()
    # i = 0
    
    MAT_FILE = os.path.join('Pz and Sz', 'measured','Pz1.mat')
    print(MAT_FILE)
    mat_contents    = sio.loadmat(MAT_FILE)
    Wc_vectors      = mat_contents['PZ1']
    Wc_vectors      = Wc_vectors.squeeze(1)
    print(Wc_vectors.shape)

# xlsx_file = os.path.join("Duct_path", "Primary Path.csv")
# print(xlsx_file)
# dfs = pd.read_csv(xlsx_file)
# print(np.array(dfs['Amplitude - Plot 0']))
    i = 0 