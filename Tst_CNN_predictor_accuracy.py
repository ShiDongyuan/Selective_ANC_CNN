import numpy as np
import torch 
from torch import nn
from torch.utils.data import DataLoader 
from MyDataLoader import MyNoiseDataset
from Tst_CNN_predicotr_v1 import Filter_ID_predictor, Filter_ID_predictor_from_1DCNN_LMSoftmax
from Train_validate import create_data_loader
#----------------------------------------------------------------
from pytorch_metric_learning import losses
from pytorch_metric_learning.utils.inference import LogitGetter
#-------------------------------------------------------------
# Function  :   load_weigth_for_model()
# Loading the weights to model from pre-trained coefficients 
#-------------------------------------------------------------
def load_weigth_for_model(model, pretrained_path, device):
    model_dict      = model.state_dict()
    pretrained_dict = torch.load(pretrained_path,map_location= device)

    for k, v in model_dict.items():
        model_dict[k] = pretrained_dict[k]
    
    model.load_state_dict(model_dict)
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
        print(f"----------------------------------------")
        print(f"The {i}-th iteration's accuracy is {acc}")
        accuracy_vec.append(acc)
    return accuracy_vec, sum(accuracy_vec)/len(accuracy_vec)

#----------------------------------------------------------------
# Function  : Testing accuracy of the predictor (coming from main)
#----------------------------------------------------------------
def Testing_model_accuracy(MODEL_PATH, MATFILE_PATH, VALIDATTION_FILE):
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using {device}")
    #
    BATCH_SIZE       = 100
    fs               = 16000
    sheet            = "Index.csv"
    CNN_classfier  = Filter_ID_predictor(MODEL_PATH, MATFILE_PATH, fs, device)
    valid_data     = MyNoiseDataset(VALIDATTION_FILE,sheet)
    valid_dataloader = create_data_loader(valid_data,BATCH_SIZE)
    _, average_acc = tst_accuracy_of_model(valid_dataloader,CNN_classfier)
    print(f"The average accuracy is {average_acc}")

#------------------------------------------------------------------------------------------
# Function  : Testing accuracy of the predictor of 1D_CNN with LMSoftmax (coming from main)
#------------------------------------------------------------------------------------------
def Testing_model_with_LMSoftmax_accuracy(MODEL_PATH,MATFILE_PATH, VALIDATTION_FILE, LMSOFTMAX_WEIGHT_PTH):
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using {device}")
    #
    BATCH_SIZE       = 100
    fs               = 16000
    sheet            = "Index.csv"
    
    
    LMSoftmax_weight =losses.LargeMarginSoftmaxLoss(num_classes= 15, embedding_size= 620, margin = 2).to(device)
    load_weigth_for_model(model=LMSoftmax_weight,pretrained_path=LMSOFTMAX_WEIGHT_PTH,device=device)
    LMSoftmax_weight.eval()
    
    CNN_classfier    = Filter_ID_predictor_from_1DCNN_LMSoftmax(MODEL_PATH, MATFILE_PATH, fs, LMSoftmax_weight, device)
    valid_data       = MyNoiseDataset(VALIDATTION_FILE,sheet)
    valid_dataloader = create_data_loader(valid_data,BATCH_SIZE)
    _, average_acc   = tst_accuracy_of_model(valid_dataloader,CNN_classfier)
    print(f"The average accuracy is {average_acc}")
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