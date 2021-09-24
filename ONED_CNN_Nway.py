import torch 
from torch import nn
from torch._C import device
from torchsummary import summary

from torch.utils.data import DataLoader 
from MyDataLoader import MyNoiseDataset

import matplotlib
import matplotlib.pyplot as plt

#--------------------------------------------------------------
# Class type    :   OneD_CNN_Nway
# Description   : This calss is used to finetune for the 
# unkown noise source. 
#--------------------------------------------------------------
class OneD_CNN_Nway(nn.Module):

    def __init__(self):
        super().__init__()
        # First layer 
        self.conv1 = nn.Conv1d(
            in_channels = 1 ,
            out_channels= 10, 
            kernel_size = 3 , 
            stride      = 1 
            )
        # Secondary layer 
        self.conv2 = nn.Sequential(
            nn.Conv1d(
                in_channels = 10 ,
                out_channels= 20 ,
                kernel_size = 32 ,
                stride      = 1         
            ),
        # Max-pool 
            nn.MaxPool1d(
                kernel_size = 512 ,
                stride      = 512 
            )
        )
        # CosSimilarity 
        self.cos = nn.CosineSimilarity(dim=1, eps= 1e-8)
    
    def forward(self, input_data):
        x = self.conv1(input_data)
        x = self.conv2(x)
        x = x.view(x.shape[0],-1)
        test_embdeeding = x[0].unsqueeze(dim=0)
        targ_embdeeding = x[1:]
        logit = self.cos(test_embdeeding, targ_embdeeding).unsqueeze(dim=0)

        return logit 

#--------------------------------------------------------------
class ONED_CNN_Nway_Loadedcoef(OneD_CNN_Nway):
    
    def __init__(self,pretrained_path,device):
        super().__init__()
        self.load_weigth_for_model(pretrained_path,device)
    
    #--------------------------------------------------------------
    # Function      :   load_weigth_for_model()
    # Description   : Loading the weights to model from pre-trained 
    # coefficients. 
    #--------------------------------------------------------------
    def load_weigth_for_model(self, pretrained_path, device):
        model_dict      = self.state_dict()
        pretrained_dict = torch.load(pretrained_path,map_location=device)

        for k, v in model_dict.items():
            model_dict[k] = pretrained_dict[k]
        
        self.load_state_dict(model_dict)
#--------------------------------------------------------------
# Function      :   create_data_loder
# Description   : This function is used to create the data 
# loader.  
#--------------------------------------------------------------
def create_data_loader(train_data, batch_size):
    train_dataloader = DataLoader(train_data, batch_size)
    return train_dataloader

#--------------------------------------------------------------
# Function      :   Main function 
#--------------------------------------------------------------

if __name__ == "__main__":

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print('----------------------------------------------------------------')
    print(f'        This program uses {device} !!!')
    print('----------------------------------------------------------------')
    CNN = OneD_CNN_Nway()

    summary(CNN, (1,16000))
    

    CNN_1 = ONED_CNN_Nway_Loadedcoef('feedforwardnet.pth',device)
    for param in CNN_1.parameters():
        print(param)
    summary(CNN_1, (1,16000))

    

    i = 0 



        
        