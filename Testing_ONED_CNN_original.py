import torch
from torch import nn
from torch.utils.data import DataLoader 
from MyDataLoader import MyNoiseDataset

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

    print(f"Validat loss : {eval_loss/i}" + f" Validat accuracy : {eval_acc/i}") 
    return eval_acc/i

#----------------------------------------------------------------------------------------
# Function : Testing the accuracy of the trained model in the testing set.
#----------------------------------------------------------------------------------------
def Test_model_accuracy_original(TESTING_DATASET_FILE=None, MODLE_CLASS=None, MODLE_PTH=None):
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
    accuracy = validate_single_epoch(feed_forward_net, testing_loader, loss_fn, device)
    
    return accuracy
    