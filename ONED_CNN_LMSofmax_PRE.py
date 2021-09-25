import torch 
from torch import nn
from torchsummary import summary

from torch.utils.data import DataLoader 
from MyDataLoader import MyNoiseDataset

import matplotlib
import matplotlib.pyplot as plt

#--------------------------------------------------------
# Class type  : OneD_CNN_Pre
# Description : The oneD cnn model is used to estimate 
# the distance between target and data.  
#--------------------------------------------------------
class OneD_CNN__LMSoftmax_Pre(nn.Module):
    
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
        # self.normalized = nn.BatchNorm1d(
        #     num_features= 620
        # )
        # # Linear layer 1
        # self.linear1 = nn.Linear(
        #     in_features = 620 , 
        #     out_features= 15
        # )
        # # active function 
        # self.active = nn.Tanh() 
        # # Linear layer 2
        # self.linear2 = nn.Linear(
        #     in_features = 15, 
        #     out_features= 15 
        # )
        # # softmax layer 
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, input_data):
        x = self.conv1(input_data)
        x = self.conv2(x)
        pre = x.view(x.shape[0],-1)
        #x = self.flat(x)
        #x = x.reshape(x.size(0), -1)
        #pre = self.normalized(x)
        # x = self.linear1(x)
        # x = self.active(x)
        # logits = self.linear2(x)
        # prediction = self.softmax(logits)

        return pre 

# Loading the weights to model from pre-trained coefficients 
def load_weigth_for_model(model, pretrained_path,device):
    model_dict      = model.state_dict()
    pretrained_dict = torch.load(pretrained_path,map_location=device)

    for k, v in model_dict.items():
        model_dict[k] = pretrained_dict[k]
    
    model.load_state_dict(model_dict)

# Cosine distance between two tensor 
def cosine_distance(x1, x2=None, eps=1e-8):
    x2 = x1 if x2 is None else x2
    w1 = x1.norm(p=2, dim=1, keepdim=True)
    w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
    return 1 - torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)

def plot_specgram(waveform, sample_rate, title="Spectrogram", xlim=None):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].specgram(waveform[c], Fs=sample_rate)
        if num_channels > 1:
            axes[c].set_ylabel(f'Channel {c+1}')
        if xlim:
            axes[c].set_xlim(xlim)
    figure.suptitle(title)
    plt.show(block=False)
#------------------------------------------------------------------------
# Class       : minmaxscaler()
# Description : Shrink the data 
#------------------------------------------------------------------------
def minmaxscaler(data):
    min = data.min()
    max = data.max()    
    return (data)/(max-min)

#------------------------------------------------------------------------
# Class type    : OneD_CNN_predictor 
# Description   : 
#------------------------------------------------------------------------
class   OneD_CNN_Predictor():

    def __init__(self, MODEL_PATH,device):
        self.cnn = OneD_CNN__LMSoftmax_Pre().to(device)
        load_weigth_for_model(self.cnn,MODEL_PATH,device)
        self.cnn.eval()
        self.cos = nn.CosineSimilarity(dim=1)
    
    def cosSimilarity(self, signal_1, signal_2):
        signal1, signal2 = signal_1.unsqueeze(0), signal_2.unsqueeze(0)
        similarity = self.cos(self.cnn(signal1), self.cnn(signal2))
        return similarity.item() 
    
    def cosSimilarity_minmax(self, signal_1, signal_2):
        signal1, signal2 = minmaxscaler(signal_1).unsqueeze(0), minmaxscaler(signal_2).unsqueeze(0)
        similarity = self.cos(self.cnn(signal1), self.cnn(signal2))
        return similarity.item() 

#------------------------------------------------------------------------
if __name__ == "__main__":
    MODEL_PATH = "feedforwardnet_LMSoftMax.pth"
    cnn = OneD_CNN__LMSoftmax_Pre() 

    load_weigth_for_model(cnn,MODEL_PATH,device='cpu')

    cnn.eval()
    
    input = torch.randn(1, 1, 16000).type(torch.float32)
    out = cnn(input)
    print(out.shape)

    d = cosine_distance(out)
    print(d)

    cos = nn.CosineSimilarity(dim=1)
    output = cos(out, out)
    print(output.item())

    VALIDATTION_FILE = "Validate_1\Index.csv"

    valid_data = MyNoiseDataset(VALIDATTION_FILE)
    signel1t, label1 = valid_data[230] #25
    signel1 = signel1t.unsqueeze(0)
    signel2t, label2 = valid_data[78]
    signel2 = signel2t.unsqueeze(0)
    print(f"Lable1: {label1} Label2 : {label2}")

    out1 =cos(cnn(signel1),cnn(signel2))
    print(out1)

    fs = 16000

    # construct model and assign it to device
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using {device}")

    plot_specgram(signel1t, fs)
    plot_specgram(signel2t, fs)

    Predictor = OneD_CNN_Predictor(MODEL_PATH,device)
    out2      = Predictor.cosSimilarity(signel1t.to(device), signel2t.to(device))
    print(f"The cos similarity is {out2:0.4f}.")


    i = 0 