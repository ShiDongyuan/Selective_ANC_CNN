import scipy.io as sio
import torch

#----------------------------------------------------------------
# Function: Load_Pretrained_fiters_to_tensor
# Description: Loading the control filters from the file 
#----------------------------------------------------------------
class Fixed_filter_controller():
    def __init__(self, MAT_FILE, fs):
        self.Wc = self.Load_Pretrained_filters_to_tensor(MAT_FILE)
        Len     = self.Wc.shape[1]
        self.fs = fs 
        self.Xd = torch.zeros(1, Len, dtype= torch.float)
        self.Current_Filter = torch.zeros(1, Len, dtype=torch.float)
    
    def noise_cancellation(self, Dis, Fx, filter_index):
        Erro = torch.zeros(Dis.shape[0])
        j    = 0 
        for ii, dis in enumerate(Dis):
            self.Xd      = torch.roll(self.Xd,1,1)
            self.Xd[0,0] = Fx[ii]
            yt           = self.Current_Filter @ self.Xd.t()
            Erro[ii]     = dis - yt
            if (ii + 1) % self.fs == 0 :
                self.Current_Filter = self.Wc[filter_index[j]]
                j += 1  
        return Erro 
        
    def Load_Pretrained_filters_to_tensor(self, MAT_FILE):
        mat_contents    = sio.loadmat(MAT_FILE)
        Wc_vectors      = mat_contents['Wc_v']
        return  torch.from_numpy(Wc_vectors).type(torch.float)
    
#-----------------------------------------------------------------------
# Function : main()
#-----------------------------------------------------------------------
def main():
    pass 

#-----------------------------------------------------------------------
if __name__ == "__main__":
    main()