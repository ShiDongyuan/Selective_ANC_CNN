from DataSet_construction_DesignBand import Generating_Dataset_as_Given_Frequencybands
from Filter_design import Boardband_Filter_Desgin_as_Given_Freqeuencybands
from Tst_CNN_predictor_accuracy import Testing_model_accuracy
import os

#---------------------------------------------------------------------------
if __name__ == '__main__':
    Folder_name_of_testing_data_set = 'Testing_dataset_of_5frequencybands' 
    Frequecy_band = [[20, 550], [450, 1200], [1000, 2700],[2500, 4500],[4400, 7980]]
    
    if not os.path.exists(Folder_name_of_testing_data_set):
        Generating_Dataset_as_Given_Frequencybands(N_sample=1000, F_bands= Frequecy_band, Folder_name= Folder_name_of_testing_data_set)
    else:
        print("Data of " + Folder_name_of_testing_data_set + 'is existed !!!')
        
    Filter_mat_name = 'Boardband_filter_from_5frequencybands.mat'
    if not os.path.exists(Filter_mat_name):
        Boardband_Filter_Desgin_as_Given_Freqeuencybands(MAT_filename=Filter_mat_name, F_bands=Frequecy_band,fs=16000)
    else:
        print("Data of " + Filter_mat_name + 'is existed !!!')
    
    MODEL_PATH = "feedforwardnet.pth"
    Testing_model_accuracy(MODEL_PATH=MODEL_PATH, MATFILE_PATH=Filter_mat_name, VALIDATTION_FILE=Folder_name_of_testing_data_set)
    