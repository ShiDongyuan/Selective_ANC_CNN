from os import write
import os
from sklearn.utils import shuffle
import torch 
from torch import nn
from torch.utils.data import DataLoader 
from MyDataLoader import MyNoiseDataset
from ONED_CNN import OneD_CNN
from sklearn.metrics import classification_report

BATCH_SIZE    = 250 
EPOCHS        = 30 
LEARNING_RATE = 0.001 

def create_data_loader(train_data, batch_size):
    train_dataloader = DataLoader(train_data, batch_size)
    return train_dataloader 

def train_single_epoch(model, data_loader, loss_fn, optimiser, device, iteration):
    train_loss = 0 
    train_acc  = 0 
    model.train() 
    for input, target in data_loader:
        input, target = input.to(device), target.to(device)

        # calculate loss
        prediction = model(input)
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

    # print(
    #     classification_report(target.cpu().numpy(), pred.detach().cpu().numpy())
    # )

    print(f"loss: {train_loss/len(data_loader)}")
    print(f"Accuracy : {train_acc / len(data_loader)}")

def validate_single_epoch(model, eva_data_loader, loss_fn, device):
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

        # Break the foor loop 
        if i == 40:
            break 

    print(f"Validat loss : {eval_loss/i}" + f" Validat accuracy : {eval_acc/i}") 

def train(model, data_loader, eva_data_loader, loss_fn, optimiser, device, epochs):
    for i in range(epochs):
        print(f"Epoch {i+1}")
        train_single_epoch(model, data_loader, loss_fn, optimiser, device, i)
        validate_single_epoch(model, eva_data_loader, loss_fn, device)
        print("----------------------------------")
    print("Finished trainning")
#----------------------------------------------------------------------------------------
# Function : Training and validating the pre-defined 1D-CNN (It comes from main function)
#----------------------------------------------------------------------------------------
def Train_validate_predefined_CNN(TRIAN_DATASET_FILE, VALIDATION_DATASET_FILE, MODEL_PTH):
    File_sheet = 'Index.csv'
    
    train_data = MyNoiseDataset(TRIAN_DATASET_FILE,File_sheet)
    valid_data = MyNoiseDataset(VALIDATION_DATASET_FILE,File_sheet)
    
    train_dataloader = create_data_loader(train_data, BATCH_SIZE)
    valid_dataloader = create_data_loader(valid_data,int(BATCH_SIZE/10))
    
    # construct model and assign it to device
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using {device}")

    feed_forward_net  = OneD_CNN().to(device)
    
    # initialise loss funtion + optimiser
    loss_fn = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(feed_forward_net.parameters(),
                                 lr=LEARNING_RATE)

    # train model
    train(feed_forward_net, train_dataloader, valid_dataloader, loss_fn, optimiser, device, EPOCHS)

    # save model
    torch.save(feed_forward_net.state_dict(), MODEL_PTH)
    print("Trained feed forward net saved at " + MODEL_PTH)
#----------------------------------------------------------------------------------------
if __name__ == "__main__":
    File_sheet = 'Index.csv'
    ANNOTATIONS_FILE = 'Training_data'#"DATA_1\Index.csv"
    VALIDATTION_FILE = "Validate_1" #\Index.csv"

    #pd.read_csv("DATA_1\Index.csv")
    train_data = MyNoiseDataset(ANNOTATIONS_FILE,File_sheet)
    valid_data = MyNoiseDataset(VALIDATTION_FILE,File_sheet)

    train_dataloader = create_data_loader(train_data, BATCH_SIZE)
    valid_dataloader = create_data_loader(valid_data,int(BATCH_SIZE/10))

    # construct model and assign it to device
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using {device}")

    feed_forward_net  = OneD_CNN().to(device)
    print(feed_forward_net)

     # initialise loss funtion + optimiser
    loss_fn = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(feed_forward_net.parameters(),
                                 lr=LEARNING_RATE)

    # train model
    train(feed_forward_net, train_dataloader, valid_dataloader, loss_fn, optimiser, device, EPOCHS)

    # save model
    torch.save(feed_forward_net.state_dict(), "feedforwardnet.pth")
    print("Trained feed forward net saved at feedforwardnet.pth")

