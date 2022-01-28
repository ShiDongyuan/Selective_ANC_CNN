#    _  _ _____ _   _   ___  ___  ___   _       _   ___
#   | \| |_   _| | | | |   \/ __|| _ \ | |     /_\ | _ )
#   | .` | | | | |_| | | |) \__ \|  _/ | |__  / _ \| _ \
#   |_|\_| |_|  \___/  |___/|___/|_|   |____|/_/ \_\___/
#
#---------------------------------------------------------------
import torch 
from torch import nn
from torchsummary import summary

from torch.utils.data import DataLoader 
from MyDataLoader import MyNoiseDataset

def create_data_loader(train_data, batch_size):
    train_dataloader = DataLoader(train_data, batch_size)
    return train_dataloader

class OneD_CNN(nn.Module):

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
         # Flatten layer 
        #self.flat = nn.Flatten()
        # Normalized layer 
        self.normalized = nn.BatchNorm1d(
            num_features= 620
        )
        # Linear layer 1
        self.linear1 = nn.Linear(
            in_features = 620 , 
            out_features= 15
        )
        # active function 
        self.active = nn.Tanh() 
        # Linear layer 2
        self.linear2 = nn.Linear(
            in_features = 15, 
            out_features= 15 
        )
        # softmax layer 
        #self.softmax = nn.Softmax(dim=1)

    def forward(self, input_data):
        x = self.conv1(input_data)
        x = self.conv2(x)
        x = x.view(x.shape[0],-1)
        #x = self.flat(x)
        #x = x.reshape(x.size(0), -1)
        x = self.normalized(x)
        x = self.linear1(x)
        x = self.active(x)
        logits = self.linear2(x)
        #prediction = self.softmax(logits)

        return logits #prediction 
    
if __name__ == "__main__":
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using {device}")


    CNN = OneD_CNN()
    
    summary(CNN, (1,16000))
    # CNN.eval()
    
    # input = torch.randn(2, 1, 16000).type(torch.float32)
    # print(input.shape)
    # C, pred = CNN(input).max(1)
    # print(C)
    # print(pred)

    # ANNOTATIONS_FILE = "DATA_1\Index.csv"

    # #pd.read_csv("DATA_1\Index.csv")
    # train_data = MyNoiseDataset(ANNOTATIONS_FILE)

    # train_dataloader = create_data_loader(train_data, 30)

    # signal,	label	=	next(iter(train_dataloader))
    # print(signal.shape)
    # CNN(signal)

    i = 1 
        

