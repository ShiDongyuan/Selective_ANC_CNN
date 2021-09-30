from DataSet_construction_v2 import Generating_Dataset_as_Given_Frequencylevels
from Filter_design import Broadband_Filter_Design_as_Given_F_levles
from Train_validate import Train_validate_predefined_CNN
from Tst_CNN_predictor_accuracy import Testing_model_accuracy
import os

#--------------------------------------------------------------------------------------------
if __name__ == '__main__':
    # System parameter configuraiton  
    level = 4
        
    # Creating Training, Vildation, and Testing Datasets.
    Folder_name_list_of_data_set =['Training_data', 'Validate_data', 'Testing_data']
    N_sample_list =[40000, 1000, 1000]
    for Folder_name_of_data_set, N_sample in zip(Folder_name_list_of_data_set,N_sample_list):
        if not os.path.exists(Folder_name_of_data_set):
            Generating_Dataset_as_Given_Frequencylevels(N_sample=N_sample
                                                        , level=level
                                                        , Folder_name=Folder_name_of_data_set)
        else:
            print(Folder_name_of_data_set + 'exists !!!')
    
    # Creating pre-traind broadband noise fiter 
    MAT_FILENAME = "Bandlimited_filter.mat"
    if not os.path.exists(MAT_FILENAME):
        Broadband_Filter_Design_as_Given_F_levles(MAT_filename=MAT_FILENAME, level=4, fs=16000)
    else:
        print(MAT_FILENAME + 'exists !!!')
        
    # Training and validating the pre-defined ONE_CNN model
    MODEL_PTH = 'feedforwardnet_v1.pth'
    if not os.path.exists(MODEL_PTH):
        Train_validate_predefined_CNN(TRIAN_DATASET_FILE=Folder_name_list_of_data_set[0]
                                      , VALIDATION_DATASET_FILE=Folder_name_list_of_data_set[1]
                                      , MODEL_PTH=MODEL_PTH)
    
    # Testing the pre-tained CNN model based cosine similarity 
    print('=======================================================')
    print('    Testing modle accuracy based similarity')
    print('=======================================================')
    Testing_model_accuracy(MODEL_PATH=MODEL_PTH
                           , MATFILE_PATH= MAT_FILENAME
                           , VALIDATTION_FILE= Folder_name_list_of_data_set[2])
    
        