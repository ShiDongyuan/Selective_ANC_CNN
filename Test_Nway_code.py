import numpy as np 
import torch
from ONED_CNN_PRE import OneD_CNN_Predictor
from ONED_CNN_Nway import ONED_CNN_Nway_Loadedcoef
from Fixed_Filter_noise_cancellation import Fxied_filters
from MyDataLoader import MyNoiseDataset
from Train_validate import BATCH_SIZE, create_data_loader

if __name__ == "__main__":
    
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using {device}")
    
    # System configuration parameter
    fs = 16000 
    
    # Model path 
    MODEL_PATH       = 'feedforwardnet.pth'
    MATFILE_PATH     = 'DesignBand_filter_v1.mat'
    VALIDATTION_FILE = "testing_data_v1"
    sheet            = "Index.csv"
    
    #Lodaing dataset 
    Test_data  = MyNoiseDataset(VALIDATTION_FILE,sheet)
    BATCH_SIZE = 1 
    Dataloader = create_data_loader(Test_data,BATCH_SIZE)
    i = 0 
    for noise, target in Dataloader:
        # i +=1 
        # print(noise.shape, target)
        if i == 40:
            break  
    
    #-----------------------------------------------------
    CNN_pre = OneD_CNN_Predictor(MODEL_PATH,device)
    Filters = Fxied_filters(MATFILE_PATH,fs)
    print(noise[0].shape)
    
    Num_pre_filters = len(Filters.Charactors)
    
    similarity_rato = []
    for i in range(Num_pre_filters):
        similarity_rato.append(CNN_pre.cosSimilarity(noise[0], Filters.Charactors[i]))
    
    print(similarity_rato)
    index = np.argmax(similarity_rato)
    
    #
    CNN_Nway_Pre = ONED_CNN_Nway_Loadedcoef(MODEL_PATH,device).to(device)
    # Construing the input vector 
    targ_input        = Filters.Charactors
    input_v = torch.cat((noise,targ_input), dim=0).to(device)
    prediction = CNN_Nway_Pre(input_v)
 
    print(prediction)
    i = 0 
    