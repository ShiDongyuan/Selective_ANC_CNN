import numpy as np
import torch 
from torch import nn
from torch.utils.data import DataLoader 
from MyDataLoader import MyNoiseDataset
from Tst_CNN_predicotr_v1 import Filter_ID_predictor
from Train_validate import create_data_loader
#-----------------------------------------------------------------
# Function    :
# Description : 
#-----------------------------------------------------------------
def tst_accuracy_of_model(tst_data_loder, model):
    accuracy_vec = []
    i            = 0 
    for input, target in tst_data_loder:
        i += 1
        batch_acc = 0
        for signal_1d, target_1d in zip(input, target):
            batch_acc +=(model.predic_ID(signal_1d)==target_1d.numpy())
        acc = batch_acc/len(target)
        print(f"The {i}-the iteration's accuracy is {acc}")
        accuracy_vec.append(acc)
    return accuracy_vec, sum(accuracy_vec)/len(accuracy_vec)

#-----------------------------------------------------------------
if __name__ == "__main__":
    
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using {device}")
    #device = "cpu"
    
    MATFILE_PATH     = 'Pre-train Control filter.mat'
    #----
    FILE_NAME_PATH   = "Bandlimited_filter.mat"
    FILE_NAME_PATH   = "DesignBand_filter_v1.mat"
    MATFILE_PATH     = FILE_NAME_PATH
    #-----
    MODEL_PATH       = "feedforwardnet.pth"
    # MODEL_PATH       = "feedforwardnet_Nway_Finetuned.pth"
    #MODEL_PATH       = 'feedforwardnet_LMSoftMax.pth'
    VALIDATTION_FILE = "testing_data"
    VALIDATTION_FILE = "testing_data_v1"
    sheet            = "Index.csv"
    BATCH_SIZE       = 100

    fs             = 16000
    CNN_classfier  = Filter_ID_predictor(MODEL_PATH, MATFILE_PATH, fs, device)
    valid_data     = MyNoiseDataset(VALIDATTION_FILE,sheet)
    valid_dataloader = create_data_loader(valid_data,BATCH_SIZE)
    _, average_acc = tst_accuracy_of_model(valid_dataloader,CNN_classfier)
    print(f"The average accuracy is {average_acc}")