from DataSet_construction_v2 import Generating_Dataset_as_Given_Frequencylevels
from Filter_design import Broadband_Filter_Design_as_Given_F_levles
from ONED_CNN import OneD_CNN
from Train_validate import Train_validate_predefined_CNN
from Testing_ONED_CNN_original import Test_model_accuracy_original
from Tst_CNN_predictor_accuracy import Testing_model_accuracy
from Bcolors import bcolors
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
    
    # Opening a file for storing the training and testing report 
    f = open('ONED_CNN-training-and-testing-report.txt', 'w')
        
    # Training and validating the pre-defined ONE_CNN model
    MODEL_PTH = 'feedforwardnet_v1.pth'
    if not os.path.exists(MODEL_PTH):
        acc_train, acc_validate = Train_validate_predefined_CNN(TRIAN_DATASET_FILE=Folder_name_list_of_data_set[0]
                                                                , VALIDATION_DATASET_FILE=Folder_name_list_of_data_set[1]
                                                                , MODEL_PTH=MODEL_PTH)
        repot = bcolors.OKGREEN  + f' The accuracy on training and testing are {acc_train} and {acc_validate}' + bcolors.ENDC
        f.write( repot + '\n')
    else:
        print( bcolors.WARNING + MODEL_PTH + ' has aready been trained !' + bcolors.ENDC)
    
    # Testing the trained CNN model based on the testing dataset. 
    print('=======================================================')
    print(bcolors.HEADER+'Testing modle accuracy based on testing dataset'+bcolors.ENDC)
    print('=======================================================')
    if os.path.exists(MODEL_PTH):
        acc =Test_model_accuracy_original(TESTING_DATASET_FILE=Folder_name_list_of_data_set[2]
                                          , MODLE_CLASS=OneD_CNN
                                          , MODLE_PTH=MODEL_PTH)
        repot = bcolors.OKGREEN  + f' The accuracy on testing datast is {acc}' + bcolors.ENDC
        print(repot)
        f.write(repot + '\n')
    else:
        print(bcolors.FAIL + MODEL_PTH + ' does not exsit !' + bcolors.ENDC)
    
    # Testing the pre-tained CNN model based cosine similarity 
    print('=======================================================')
    print('    Testing modle accuracy based on similarity')
    print('=======================================================')
    acc_similarity = Testing_model_accuracy(MODEL_PATH=MODEL_PTH
                                            , MATFILE_PATH= MAT_FILENAME
                                            , VALIDATTION_FILE= Folder_name_list_of_data_set[2])
    repot = bcolors.OKGREEN  + f' The accuracy on similarity is {acc_similarity}' + bcolors.ENDC
    f.write(repot + '\n')
    f.close()
    
        