from ONED_CNN_PRE import OneD_CNN_Pre
import torch
from torch import nn
#----------------------------------------------------------------------------
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

#-------------------------------------------------------------
# Class : ONE_CNN_LMSoftmax_Pre
# Construting the predictor model from ONED_CNN_PRE.py
#-------------------------------------------------------------
class ONE_CNN_LMSoftmax_Predictor():
    
    def __init__(self, MODEL_PATH, Wc, device):
        """
        :param MODEL_PATH: The pre-trained CNN model.
        :param Wc        : The weights of the LMSoftmax layer has the dimension of [embedding_size x num_classes].
        :param device    : 'cpu' or 'cuda'.
        """
        self.cnn = OneD_CNN_Pre().to(device)
        load_weigth_for_model(self.cnn,MODEL_PATH,device)
        self.cnn.eval()
        self.cos = nn.CosineSimilarity(dim=1).to(device)
        self.LG = LogitGetter(Wc)
        self.device = device
    
    def _get_embedding_(self, input):
        input_v = self.cnn(input)
        embedding = self.LG(input_v)
        return embedding 
    
    def cosSimilarity(self, signal_1, signal_2):
        """
        :param signal_1: The signal tensor has dimension of [1 x fs samples].
        :param signal_2: The signal tensor has dimension of [1 x fs samples].
        """
        signal1, signal2 = signal_1.unsqueeze(0), signal_2.unsqueeze(0)
        similarity = self.cos(self._get_embedding_(signal1.to(self.device)), self._get_embedding_(signal2.to(self.device)))
        return similarity.cpu().item()
     
    #-------------------------------------------------------------