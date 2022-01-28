#   _   _ _____ _   _   ____  ____  ____    _          _
#  | \ | |_   _| | | | |  _ \/ ___||  _ \  | |    __ _| |__
#  |  \| | | | | | | | | | | \___ \| |_) | | |   / _` | '_ \
#  | |\  | | | | |_| | | |_| |___) |  __/  | |__| (_| | |_) |
#  |_| \_| |_|  \___/  |____/|____/|_|     |_____\__,_|_.__/
#
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader 
from MyDataLoader import MyNoiseDataset

from Bcolors import bcolors
from sklearn.metrics import classification_report

BATCH_SIZE    = 250 
 
#------------------------------------------------------------
def create_data_loader(train_data, batch_size):
    train_dataloader = DataLoader(train_data, batch_size)
    return train_dataloader 

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

#-------------------------------------------------------------
# Function : validate_single_epoch()
# Testing the accuracy of the trained modle in test dataset.
#-------------------------------------------------------------
def validate_single_epoch(model, eva_data_loader, loss_fn, device, return_prdict_vector=None):
    if return_prdict_vector == None:
        eval_loss = 0 
        eval_acc  = 0 
        model.eval()
        i         = 0 
        for input, target in eva_data_loader:
            input, target = input.to(device), target.to(device)
            i            += 1 
            
            # Calculating the loss value
            prediction = model(input)
            loss       = loss_fn(prediction,target)

            # recording the validating loss and accuratcy
            eval_loss  += loss.item()
            _, pred     = prediction.max(1)
            num_correct = (pred == target).sum().item()
            acc         = num_correct / input.shape[0]
            eval_acc    += acc

        print(f"Validat loss : {eval_loss/i}" + f" Validat accuracy : {eval_acc/i}") 
        return eval_acc/i
    else:
        eval_loss = 0 
        eval_acc  = 0 
        model.eval()
        i         = 0 

        for input, target in eva_data_loader:
            input, target = input.to(device), target.to(device)
            i            += 1 
            
            # Calculating the loss value
            prediction = model(input)
            loss       = loss_fn(prediction,target)

            # recording the validating loss and accuratcy
            eval_loss  += loss.item()
            _, pred     = prediction.max(1)
            num_correct = (pred == target).sum().item()
            acc         = num_correct / input.shape[0]
            eval_acc    += acc
            
            if i == 1:
                prdic_vector = torch.clone(pred.cpu())
                targe_vector = torch.clone(target.cpu())
            else:
                prdic_vector = torch.cat((prdic_vector,pred.cpu()))
                targe_vector = torch.cat((targe_vector,target.cpu()))

        print(f"Validat loss : {eval_loss/i}" + f" Validat accuracy : {eval_acc/i}") 
        return eval_acc/i, prdic_vector, targe_vector

#----------------------------------------------------------------------------------------
# Function : Testing the accuracy of the trained model in the testing set.
#----------------------------------------------------------------------------------------
def Test_model_accuracy_original(TESTING_DATASET_FILE=None, MODLE_CLASS=None, MODLE_PTH=None, Report =None):
    File_sheet = 'Index.csv'
    
    testing_dataset = MyNoiseDataset(TESTING_DATASET_FILE, File_sheet)
    testing_loader  = create_data_loader(testing_dataset,int(BATCH_SIZE/10))
    
     # construct model and assign it to device
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using {device}")

    feed_forward_net  = MODLE_CLASS().to(device)
    
    # loading coefficients 
    load_weigth_for_model(model=feed_forward_net, pretrained_path=MODLE_PTH,device=device)
    
    # testing model
    loss_fn = nn.CrossEntropyLoss()
    if Report == None:
        accuracy = validate_single_epoch(feed_forward_net, testing_loader, loss_fn, device)
    else:
        accuracy, y_pred, y_true = validate_single_epoch(feed_forward_net, testing_loader, loss_fn, device, return_prdict_vector=True)
        target_names = ['A0', 'B0', 'B1','C0', 'C1', 'C2', 'C3','D0', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7']
        print(bcolors.RED + '<<====================Classification report=========================>>' + bcolors.ENDC)
        print(classification_report(y_true, y_pred, target_names=target_names))
        print(bcolors.RED +'<<==========================End=====================================>>' + bcolors.ENDC)
    return accuracy
    