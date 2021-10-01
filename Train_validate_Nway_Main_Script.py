from DataSet_construction_DesignBand import Generating_Dataset_as_Given_Frequencybands
from Train_validate_Nway import Train_validate_Nway_main
from Filter_design import Boardband_Filter_Desgin_as_Given_Freqeuencybands
from Tst_CNN_predictor_accuracy import Testing_model_accuracy
from Bcolors import bcolors
import os 

#-------------------------------------------------------------------------------------
if __name__ == '__main__':
    
    # Creating training, validation, and testing dataset
    Frequecy_band = [[20, 550], [450, 1200], [1000, 2700],[2500, 4500],[4400, 7980]]
    File_name_of_dataset_list = ['Train_dataset_of_5frequencybands'
                                 , 'Validate_dataset_of_5frequencybands'
                                 , 'Testing_dataset_of_5frequencybands']
    N_sample_list = [40000, 1000, 1000]
    
    for File_name_of_dataset, N_sample in zip(File_name_of_dataset_list, N_sample_list):
        if not os.path.exists(File_name_of_dataset):
            Generating_Dataset_as_Given_Frequencybands(N_sample=N_sample, F_bands=Frequecy_band, Folder_name= File_name_of_dataset) 
        else:
            print(bcolors.WARNING + File_name_of_dataset + ' exists !!!' + bcolors.ENDC)
    
    # Creating the pre-trained band filter for 5 different frequency band    
    Filter_mat_name = 'Boardband_filter_from_5frequencybands.mat'
    if not os.path.exists(Filter_mat_name):
        Boardband_Filter_Desgin_as_Given_Freqeuencybands(MAT_filename=Filter_mat_name, F_bands=Frequecy_band,fs=16000)
    else:
        print("Data of " + Filter_mat_name + ' is existed !!!')  
    
    # Fine-tuning the 1D CNN model based on N-way
    print('=======================================================')
    print(bcolors.OKGREEN + '    Fine-tuning the 1D CNN model based on N-way-M-shot' + bcolors.ENDC)
    print('=======================================================')
    Pre_net_model = 'feedforwardnet_LMSoftmax_v4.pth'#"feedforwardnet_v1.pth"
    MODEL_PTH     = "feedforwardnet_Nway_v1.pth"
    if not os.path.exists(MODEL_PTH):
        Train_validate_Nway_main(PRETRAINED_MODEL=Pre_net_model
                                 , FILE_NAME_PATH=Filter_mat_name
                                 , TRAININ_DATA_FILE=File_name_of_dataset_list[1]
                                 , MODEL_PTH=MODEL_PTH)
    else:
        print(bcolors.WARNING + MODEL_PTH + ' exists !!!' + bcolors.ENDC)   
        
    # Testing the accuracy of ONED_CNN_LMSoftmax
    print('=======================================================')
    print(bcolors.HEADER + '    Testing modle accuracy based on LMSoftmax' + bcolors.ENDC)
    print('=======================================================')
    Testing_model_accuracy(MODEL_PATH=MODEL_PTH
                           , MATFILE_PATH=Filter_mat_name
                           , VALIDATTION_FILE=File_name_of_dataset_list[2])                    