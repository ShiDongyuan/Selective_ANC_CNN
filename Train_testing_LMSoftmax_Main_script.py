from DataSet_construction_DesignBand import Generating_Dataset_as_Given_Frequencybands
from Filter_design import Boardband_Filter_Desgin_as_Given_Freqeuencybands
from Train_validate_LargeMarginSoftMaxLoss import Training_predefined_model_by_LargMarginSoftMaxLoss
import os

# Mian function: This seciton is used to runing the script of the code 
if __name__ == '__main__':
    Folder_name_of_train_data_set = 'Train_dataset_of_5frequencybands' 
    Frequecy_band = [[20, 550], [450, 1200], [1000, 2700],[2500, 4500],[4400, 7980]]
    
    # Creating training dataset 
    if not os.path.exists(Folder_name_of_train_data_set):
        Generating_Dataset_as_Given_Frequencybands(N_sample=40000, F_bands= Frequecy_band, Folder_name= Folder_name_of_train_data_set)
    else:
        print("Data of " + Folder_name_of_train_data_set + ' is existed !!!')
    
    # Creating validate dataset
    Folder_name_of_validate_data_set = 'Validate_dataset_of_5frequencybands'
    if not os.path.exists(Folder_name_of_validate_data_set):
        Generating_Dataset_as_Given_Frequencybands(N_sample=1000, F_bands= Frequecy_band, Folder_name= Folder_name_of_validate_data_set)
    else:
        print("Data of " + Folder_name_of_validate_data_set + ' is existed !!!') 
    
    # Creating testing data set 
    Folder_name_of_testing_data_set = 'Testing_dataset_of_5frequencybands'    
    if not os.path.exists(Folder_name_of_testing_data_set):
        Generating_Dataset_as_Given_Frequencybands(N_sample=1000, F_bands= Frequecy_band, Folder_name= Folder_name_of_testing_data_set)
    else:
        print("Data of " + Folder_name_of_testing_data_set + ' is existed !!!')
    
    # Creating the pre-trained band filter    
    Filter_mat_name = 'Boardband_filter_from_5frequencybands.mat'
    if not os.path.exists(Filter_mat_name):
        Boardband_Filter_Desgin_as_Given_Freqeuencybands(MAT_filename=Filter_mat_name, F_bands=Frequecy_band,fs=16000)
    else:
        print("Data of " + Filter_mat_name + ' is existed !!!')
        
    # Training the ONED_CNN_LMSofmax 
    Model_pth = 'feedforwardnet_LMSoftmax_for5frequencybands.pth'
    Training_predefined_model_by_LargMarginSoftMaxLoss(TRAINNING_DATA=Folder_name_of_train_data_set,
                                                       VALIDATION_DATA=Folder_name_of_validate_data_set,
                                                       MODEL_Pth=Model_pth)