from DataSet_construction_DesignBand import Generating_Dataset_as_Given_Frequencybands
from DataSet_construction_v2 import Generating_Dataset_as_Given_Frequencylevels
from Filter_design import Boardband_Filter_Desgin_as_Given_Freqeuencybands
from Train_validate_LargeMarginSoftMaxLoss import Training_predefined_model_by_LargMarginSoftMaxLoss
from Bcolors import bcolors
from Tst_CNN_predictor_accuracy import Testing_model_with_LMSoftmax_accuracy,Testing_model_accuracy
from ONED_CNN_LMSoftmax_1FC import OneD_CNN_LMSoftmax_1FC
import os

# Mian function: This seciton is used to runing the script of the code 
if __name__ == '__main__':
    Folder_name_of_train_data_set = 'Train_dataset_of_5frequencybands' 
    Frequecy_band = [[20, 550], [450, 1200], [1000, 2700],[2500, 4500],[4400, 7980]]
    
    # Creating training dataset for 5 different frequency band
    if not os.path.exists(Folder_name_of_train_data_set):
        Generating_Dataset_as_Given_Frequencybands(N_sample=40000, F_bands= Frequecy_band, Folder_name= Folder_name_of_train_data_set)
    else:
        print("Data of " + Folder_name_of_train_data_set + ' is existed !!!')
    
    # Creating validate dataset for 5 different frequency band
    Folder_name_of_validate_data_set = 'Validate_dataset_of_5frequencybands'
    if not os.path.exists(Folder_name_of_validate_data_set):
        Generating_Dataset_as_Given_Frequencybands(N_sample=1000, F_bands= Frequecy_band, Folder_name= Folder_name_of_validate_data_set)
    else:
        print("Data of " + Folder_name_of_validate_data_set + ' is existed !!!') 
    
    # Creating testing data set for 5 different frequency band
    Folder_name_of_testing_data_set = 'Testing_dataset_of_5frequencybands'    
    if not os.path.exists(Folder_name_of_testing_data_set):
        Generating_Dataset_as_Given_Frequencybands(N_sample=1000, F_bands= Frequecy_band, Folder_name= Folder_name_of_testing_data_set)
    else:
        print("Data of " + Folder_name_of_testing_data_set + ' is existed !!!')
    
    # Creating the pre-trained band filter for 5 different frequency band    
    Filter_mat_name = 'Boardband_filter_from_5frequencybands.mat'
    if not os.path.exists(Filter_mat_name):
        Boardband_Filter_Desgin_as_Given_Freqeuencybands(MAT_filename=Filter_mat_name, F_bands=Frequecy_band,fs=16000)
    else:
        print("Data of " + Filter_mat_name + ' is existed !!!')
    
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
        
    # Training the ONED_CNN_LMSofmax 
    print('=======================================================')
    print(bcolors.OKGREEN + '    Training modle accuracy based similarity' + bcolors.ENDC)
    print('=======================================================')
    Model_pth = 'feedforwardnet_LMSoftmax_v4.pth'
    Weight_pth = 'LMSoftmax_layer_weiths.pth'
    CNN_LMSoftmax_1FC = OneD_CNN_LMSoftmax_1FC()
    if not os.path.exists(Model_pth):
        Training_predefined_model_by_LargMarginSoftMaxLoss(MODEL_STRUCTURE= CNN_LMSoftmax_1FC
                                                           , TRAINNING_DATA=Folder_name_list_of_data_set[0]
                                                           , VALIDATION_DATA= Folder_name_list_of_data_set[1]
                                                           , MODEL_Pth= Model_pth
                                                           , WEIGHT_Pth= Weight_pth)
    else:
        print(bcolors.WARNING + Model_pth + ' has areadly been trained !!!' + bcolors.ENDC)
        
    # Testing the accuracy of ONED_CNN_LMSoftmax
    print('=======================================================')
    print(bcolors.HEADER + '    Testing modle accuracy based on LMSoftmax' + bcolors.ENDC)
    print('=======================================================')
    Testing_model_accuracy(MODEL_PATH=Model_pth
                                        , MATFILE_PATH=Filter_mat_name
                                        , VALIDATTION_FILE=Folder_name_of_testing_data_set)
                                        #, LMSOFTMAX_WEIGHT_PTH=Weight_pth)