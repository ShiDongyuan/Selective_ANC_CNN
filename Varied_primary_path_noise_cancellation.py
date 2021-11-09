from varied_primary_path import varid_primary_path_distrubance
from Reading_path_tst import loading_paths_from_MAT
import matplotlib.pyplot as plt 

from scipy.io import savemat
from Control_filter_selection import Control_filter_selection
from FxLMS_algorithm import FxLMS_algroithm, train_fxlms_algorithm
import numpy as np 
from Fixed_filter_noise_cancellation_v1 import Fixed_filter_controller

mdict = {}

f_vector=[600, 1200]
fs       = 16000 
Pri_path, Secon_path = loading_paths_from_MAT()
Dis, Fx, Re, Pri_vector = varid_primary_path_distrubance(fs=fs, T=9, f_vector=[600, 1200], Pri_path=Pri_path, Sec_path= Secon_path, SNR =[120, 30, 10])

mdict['Dis']  = Dis.numpy()
mdict['Path'] = Pri_vector
#=============================================================================================
# the simulaitons of the FxLMS algorithm 
controller = FxLMS_algroithm(Len=256)
Erro = train_fxlms_algorithm(Model=controller,Ref=Fx, Disturbance=Dis, Stepsize = 0.000003)

# Drawing the impulse response of the primary path
plt.title('The response of the primary path')
Time = np.arange(len(Erro))*(1/fs)
plt.plot(Time, Dis,Time, Erro)
plt.ylabel('Amplitude')
plt.xlabel('Time')
plt.grid()
plt.show()

mdict['Erro_FxLMS']= Erro

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

mdict['Erro_type_I'] = ErroC.numpy()

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

mdict['Erro_type_II']= ErroC.numpy()

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

mdict['Erro_type_III']= ErroC.numpy()

# Saving the mat data for Matlab analysis 
FILE_NAME_PATH = 'Varied_primary_path_noise_cancellation_drawing.mat'
savemat(FILE_NAME_PATH, mdict)