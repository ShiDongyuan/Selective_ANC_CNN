from Disturbance_generation import Varied_distrubance_reference_generation_from_Fvector
from Reading_path_tst import loading_paths_from_MAT
import matplotlib.pyplot as plt 
from Control_filter_selection import Control_filter_selection
from FxLMS_algorithm import FxLMS_algroithm, train_fxlms_algorithm
import numpy as np 
from Fixed_filter_noise_cancellation_v1 import Fixed_filter_controller

from scipy.io import savemat

F_vector = [[100, 500], [890, 1200], [1500, 2500]]
fs       = 16000 
Pri_path, Secon_path = loading_paths_from_MAT()
Dis, Fx, Re = Varied_distrubance_reference_generation_from_Fvector(fs=fs, T=9, f_vector=F_vector, Pri_path=Pri_path, Sec_path=Secon_path)
print(Re.shape[0]/fs)

mdict= {'Dis': Dis}

plt.title('The response of the primary path')
plt.plot(Re.numpy())
plt.ylabel('Amplitude')
plt.xlabel('Time')
plt.grid()
plt.show()

#=============================================================================================
# the simulaitons of the FxLMS algorithm 
controller = FxLMS_algroithm(Len=256)
Erro = train_fxlms_algorithm(Model=controller,Ref=Fx, Disturbance=Dis, Stepsize = 0.00000001)

# Drawing the impulse response of the primary path
plt.title('The response of the primary path')
Time = np.arange(len(Erro))*(1/fs)
plt.plot(Time, Dis,Time, Erro)
plt.ylabel('Amplitude')
plt.xlabel('Time')
plt.grid()
plt.show()

mdict= {'Erro_FxLMS': Erro}

#=============================================================================================
# 1D CNN 
id_vector = Control_filter_selection(MODEL_PTH_type=0, fs=16000,Primary_noise=Re.unsqueeze(0))
print(id_vector)

FILE_NAME_PATH     = 'Control_filter_from_5frequencies.mat'
Fixed_Cancellation = Fixed_filter_controller(MAT_FILE=FILE_NAME_PATH, fs=16000)
ErroC = Fixed_Cancellation.noise_cancellation(Dis=Dis, Fx=Fx, filter_index=id_vector)

# Drawing the impulse response of the primary path
plt.title('The response of the primary path')
Time = np.arange(len(Erro))*(1/fs)
plt.plot(Time, Dis,Time, ErroC)
plt.ylabel('Amplitude')
plt.xlabel('Time')
plt.grid()
plt.show()

mdict= {'Erro_tpye_I': Erro}

#=============================================================================================
# 1D CNN + LMSoftmax loss
id_vector = Control_filter_selection(MODEL_PTH_type=1, fs=16000,Primary_noise=Re.unsqueeze(0))
print(id_vector)

FILE_NAME_PATH     = 'Control_filter_from_5frequencies.mat'
Fixed_Cancellation = Fixed_filter_controller(MAT_FILE=FILE_NAME_PATH, fs=16000)
ErroC = Fixed_Cancellation.noise_cancellation(Dis=Dis, Fx=Fx, filter_index=id_vector)

# Drawing the impulse response of the primary path
plt.title('The response of the primary path')
Time = np.arange(len(Erro))*(1/fs)
plt.plot(Time, Dis,Time, ErroC)
plt.ylabel('Amplitude')
plt.xlabel('Time')
plt.grid()
plt.show()

mdict= {'Erro_tpye_II': Erro}

#=============================================================================================
# N shot learning 
id_vector = Control_filter_selection(MODEL_PTH_type=2, fs=16000,Primary_noise=Re.unsqueeze(0))
print(id_vector)

FILE_NAME_PATH     = 'Control_filter_from_5frequencies.mat'
Fixed_Cancellation = Fixed_filter_controller(MAT_FILE=FILE_NAME_PATH, fs=16000)
ErroC = Fixed_Cancellation.noise_cancellation(Dis=Dis, Fx=Fx, filter_index=id_vector)

# Drawing the impulse response of the primary path
plt.title('The response of the primary path')
Time = np.arange(len(Erro))*(1/fs)
plt.plot(Time, Dis,Time, ErroC)
plt.ylabel('Amplitude')
plt.xlabel('Time')
plt.grid()
plt.show()

mdict= {'Erro_tpye_III': Erro}

# Saving the mat data for Matlab analysis 
FILE_NAME_PATH = 'Varied_broadband_noise_cancellation_drawing.mat'
savemat(FILE_NAME_PATH, mdict)