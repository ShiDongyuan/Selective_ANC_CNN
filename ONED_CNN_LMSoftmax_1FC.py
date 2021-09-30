import torch 
from torch import nn
from torchsummary import summary

from torch.utils.data import DataLoader 
from MyDataLoader import MyNoiseDataset

def create_data_loader(train_data, batch_size):
    train_dataloader = DataLoader(train_data, batch_size)
    return train_dataloader

class OneD_CNN_LMSoftmax_1FC(nn.Module):

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
        # Normalized layer 
        self.normalized = nn.BatchNorm1d(
            num_features= 620
        )

        self.active = nn.Tanh() 

    def forward(self, input_data):
        x = self.conv1(input_data)
        x = self.conv2(x)
        x = x.view(x.shape[0],-1)
        x = self.normalized(x)
        logits = self.active(x)

        return logits 
    
if __name__ == "__main__":
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using {device}")


    CNN = OneD_CNN_LMSoftmax_1FC()
    
    summary(CNN, (1,16000))

    i = 1 
        

