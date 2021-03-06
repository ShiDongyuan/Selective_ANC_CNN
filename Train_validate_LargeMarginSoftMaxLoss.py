from os import write
from sklearn.utils import shuffle
import torch 
from torch import nn
from torch.utils.data import DataLoader 
from MyDataLoader import MyNoiseDataset
from ONED_CNN_LMSoftmax import OneD_CNN_LMSoftmax
from sklearn.metrics import classification_report
from Bcolors import bcolors
#----------------------------------------------------------------------------

from pytorch_metric_learning import losses
from pytorch_metric_learning.utils.inference import LogitGetter
#----------------------------------------------------------------------------
BATCH_SIZE    = 250 
EPOCHS        = 100 # 30 
LEARNING_RATE = 0.03#0.001 

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
        LG = LogitGetter(loss_fn)
        train_loss += loss.item()
        prediction  = LG(prediction)
        _, pred     = prediction.max(1)
        num_correct =(pred == target).sum().item()
        acc         = num_correct / input.shape[0]
        train_acc  += acc 

    # print(
    #     classification_report(target.cpu().numpy(), pred.detach().cpu().numpy())
    # )
    
    print(f"loss: {train_loss/len(data_loader)}")
    print(f"Accuracy : {train_acc / len(data_loader)}")
    return LG, train_acc / len(data_loader)

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
        prediction       = loss_fn(prediction)

        # recording the validating loss and accuratcy
        eval_loss  += 0
        _, pred     = prediction.max(1)
        num_correct = (pred == target).sum().item()
        acc         = num_correct / input.shape[0]
        eval_acc    += acc

        # Break the foor loop 
        if i == 40:
            break 

    print(f"Validat loss : {eval_loss/i}" + f" Validat accuracy : {eval_acc/i}") 
    return eval_acc/i

def train(model, data_loader, eva_data_loader, loss_fn, optimiser, device, epochs, scheduler2, MODEL_Pth):
    acc_past_max = 0 
    for i in range(epochs):
        print(f"Epoch {i+1}")
        LG, acc  = train_single_epoch(model, data_loader, loss_fn, optimiser, device, i)
        scheduler2.step()
        validate_single_epoch(model, eva_data_loader, LG, device)
        
        # Saving the best classifier model 
        if acc_past_max < acc:
            acc_past_max = acc
            torch.save(model.state_dict(), MODEL_Pth)
            print(bcolors.OKCYAN + "Trained feed forward net saved at " + MODEL_Pth + bcolors.ENDC) 
              
        print("----------------------------------")
    print("Finished trainning")

#---------------------------------------------------------------------------------
# Function : Training_predefined_model_by_LargMarginSoftMaxLoss (Coming from main)
#---------------------------------------------------------------------------------
def Training_predefined_model_by_LargMarginSoftMaxLoss( MODEL_STRUCTURE=None, TRAINNING_DATA=None, VALIDATION_DATA=None, MODEL_Pth=None, WEIGHT_Pth = None):
    File_sheet = "Index.csv"
    
    train_data = MyNoiseDataset(TRAINNING_DATA, File_sheet)
    valid_data = MyNoiseDataset(VALIDATION_DATA, File_sheet)
    
    train_dataloader = create_data_loader(train_data, BATCH_SIZE)
    valid_dataloader = create_data_loader(valid_data,int(BATCH_SIZE/10))
    
    # construct model and assign it to device
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using {device}")
    
    if MODEL_STRUCTURE :
        feed_forward_net  = MODEL_STRUCTURE.to(device)
        embedding_size = 620
        print(bcolors.OKCYAN + 'Using user defined CNN model.--->'+bcolors.ENDC)
    else:
        feed_forward_net  = OneD_CNN_LMSoftmax().to(device)
        embedding_size = 15
        print(bcolors.OKCYAN + 'Using program default CNN model.--->'+bcolors.ENDC)
    loss_fn = losses.LargeMarginSoftmaxLoss(num_classes= 15, embedding_size= embedding_size, margin = 2).to(device)
    optimiser = torch.optim.Adam([{'params': feed_forward_net.parameters()},
                                  {'params': loss_fn.parameters()}
                                ],
                                 lr=LEARNING_RATE)
    
    # Scheduler for the training progress 
    scheduler2 = torch.optim.lr_scheduler.MultiStepLR(optimiser, milestones=[30,80], gamma=0.1)

    # train model
    train(feed_forward_net, train_dataloader, valid_dataloader, loss_fn, optimiser, device, EPOCHS, scheduler2, MODEL_Pth)

    # save model
    # torch.save(feed_forward_net.state_dict(), MODEL_Pth)
    # print("Trained feed forward net saved at " + MODEL_Pth)
    
    if WEIGHT_Pth:
        torch.save(loss_fn.state_dict(), WEIGHT_Pth)
        print('Saved the weights of the LMSoftmax layer')
    else:
        print('Not saving the weights of the LMSoftmax layer')
        
    
#---------------------------------------------------------------------------------
if __name__ == "__main__":
    # ANNOTATIONS_FILE = "DATA_1\Index.csv"
    # VALIDATTION_FILE = "Validate_1\Index.csv"
    TRAINNING_DATA = "Training_data"
    VALIDATION_DATA = "Validate_1"
    File_sheet = "Index.csv"

    #pd.read_csv("DATA_1\Index.csv")
    train_data = MyNoiseDataset(TRAINNING_DATA, File_sheet)
    valid_data = MyNoiseDataset(VALIDATION_DATA, File_sheet)

    train_dataloader = create_data_loader(train_data, BATCH_SIZE)
    valid_dataloader = create_data_loader(valid_data,int(BATCH_SIZE/10))

    # construct model and assign it to device
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using {device}")

    feed_forward_net  = OneD_CNN_LMSoftmax().to(device)
    print(feed_forward_net)

     # initialise loss funtion + optimiser
    # loss_fn = nn.CrossEntropyLoss()
    loss_fn = losses.LargeMarginSoftmaxLoss(num_classes= 15, embedding_size= 15, margin = 3).to(device)
    optimiser = torch.optim.Adam([{'params': feed_forward_net.parameters()},
                                  {'params': loss_fn.parameters()}
                                ],
                                 lr=LEARNING_RATE)

    # train model
    train(feed_forward_net, train_dataloader, valid_dataloader, loss_fn, optimiser, device, EPOCHS)

    # save model
    torch.save(feed_forward_net.state_dict(), "feedforwardnet_LMSoftMax.pth")
    print("Trained feed forward net saved at feedforwardnet.pth")

