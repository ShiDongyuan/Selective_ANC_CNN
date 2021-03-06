#from Train_validate import File_sheet
from os import write
from sklearn.utils import shuffle
import torch 
from torch import nn
from torch.utils.data import DataLoader
# User libary  
from MyDataLoader import MyNoiseDataset
from ONED_CNN_Nway import ONED_CNN_Nway_Loadedcoef
from sklearn.metrics import classification_report
from Fixed_Filter_noise_cancellation import Fxied_filters
from Bcolors import bcolors

import time
#--------------------------------------------------------------
# Gobal parameter 
#--------------------------------------------------------------
BATCH_SIZE    = 1 
EPOCHS        = 50 #50
LEARNING_RATE = 0.00006 #0.001 0.002 0.0005 0.00006 0.00001

#--------------------------------------------------------------
# Function      :   create_data_loader()
# Description   : Creating the data loader  
#--------------------------------------------------------------
def create_data_loader(train_data, batch_size):
    train_dataloader = DataLoader(train_data, batch_size,shuffle=True)
    return train_dataloader 


#--------------------------------------------------------------
# Function      :   train_single_epoch()
# Description   : trainning the model at one poch  
#--------------------------------------------------------------
def train_single_epoch(model, data_loader, loss_fn, optimiser, device,targ_input):
    train_loss = 0 
    train_acc  = 0 
    model.train() 
    for input, target in data_loader:
        input, target = input, target.to(device)
        
        # Construing the input vector 
        input_v = torch.cat((input,targ_input), dim=0).to(device)

        # calculate loss
        prediction = model(input_v)
        loss       = loss_fn(prediction,target)

        # backpropagate error and update weights, 
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        
        # Recording the loss and accuracy 
        train_loss += loss.item()
        _, pred     = prediction.max(1)
        num_correct =(pred == target).sum().item()
        acc         = num_correct / input.shape[0]
        train_acc  += acc 

    print(f"Loss: {train_loss/len(data_loader)}")
    print(f"Accuracy : {train_acc / len(data_loader)}")
    return train_acc/len(data_loader)
    
#--------------------------------------------------------------
# Function      :   train()
# Description   : trainning the model at one poch  
#--------------------------------------------------------------
def train(model, data_loader, loss_fn, optimiser, device, targ_input, epochs, MODEL_PTH=None):
    print('----------------------------------------------------------------')
    acc_max = 0
    acc_max_epoch = 0
    for i in range(epochs):
        since = time.time()
        print(f"Epoch {i+1}")
        acc = train_single_epoch(model, data_loader, loss_fn, optimiser, device, targ_input)
        time_elapsed = time.time()-since 
        if acc > acc_max :
            acc_max = acc 
            acc_max_epoch = i+1
            torch.save(model.state_dict(), MODEL_PTH) if MODEL_PTH else None
            print(bcolors.OKCYAN+'Current mode has been saved !!'+bcolors.ENDC)   if MODEL_PTH else None
        print(f'Traning complete in {time_elapsed//60:.0f}m {time_elapsed%60:.0f}s')
        print('----------------------------------------------------------------')
    print("Finished trainning.")
    print(f'Maximum accuracy is {acc_max} at the {acc_max_epoch}th epoch.')
    return True if MODEL_PTH else False

#---------------------------------------------------------------------------
# Function : training and validating the n-way-m-shot cnn (coming from main)
#----------------------------------------------------------------------------
def Train_validate_Nway_main(PRETRAINED_MODEL=None, FILE_NAME_PATH=None, TRAININ_DATA_FILE=None, MODEL_PTH=None):
    # System parameter configuration
    FILE_sheet = "Index.csv"
    fs         = 16000
    
    # construct model and assign it to device
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using {device}")
    
    # Construing data loader
    fine_tuning_data = MyNoiseDataset(TRAININ_DATA_FILE,FILE_sheet)
    train_dataloader = create_data_loader(fine_tuning_data,BATCH_SIZE)
    
    # Creating the NN model 
    feed_forward_net = ONED_CNN_Nway_Loadedcoef(PRETRAINED_MODEL,device).to(device)
    
    # Creating targ_input of the pre-trained control filters.
    pre_train_filters = Fxied_filters(FILE_NAME_PATH,fs)
    targ_input        = pre_train_filters.Charactors
    
    # Initiating loss function and optimiser
    loss_fn = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(feed_forward_net.parameters(),
                                 lr=LEARNING_RATE)
    
    # Training model
    HasSaved=train(feed_forward_net, train_dataloader, loss_fn, optimiser, device, targ_input, EPOCHS, MODEL_PTH)
    
    # Saving model
    torch.save(feed_forward_net.state_dict(), MODEL_PTH) if not HasSaved else None
    print("Trained feed forward net saved at " + MODEL_PTH)
    
#--------------------------------------------------------------
# Main function 
#--------------------------------------------------------------
if __name__== "__main__":
    # String constant 
    PRETRAINED_MODEL = "feedforwardnet.pth"
    FILE_NAME_PATH   = "DesignBand_filter_v1.mat"
    VALIDATTION_FILE = "testing_data_v1"
    FILE_sheet       = "Index.csv"
    fs               = 16000 
    
    # construct model and assign it to device
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using {device}")
    
    # Construing data loader
    fine_tuning_data = MyNoiseDataset(VALIDATTION_FILE,FILE_sheet)
    train_dataloader = create_data_loader(fine_tuning_data,BATCH_SIZE)
    
    # Creating the NN model 
    feed_forward_net = ONED_CNN_Nway_Loadedcoef(PRETRAINED_MODEL,device).to(device)
    
    # Creating targ_input of the pre-trained control filters.
    pre_train_filters = Fxied_filters(FILE_NAME_PATH,fs)
    targ_input        = pre_train_filters.Charactors
    
    # Initiating loss function and optimiser
    loss_fn = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(feed_forward_net.parameters(),
                                 lr=LEARNING_RATE)
                                 
    # Training model
    train(feed_forward_net, train_dataloader, loss_fn, optimiser, device, targ_input, EPOCHS)
    
    # Saving model
    MODEL_PTH = "feedforwardnet_Nway_Finetuned.pth"
    torch.save(feed_forward_net.state_dict(), MODEL_PTH)
    print("Trained feed forward net saved at " + MODEL_PTH)