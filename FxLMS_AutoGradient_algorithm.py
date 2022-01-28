import torch
import numpy as np 
import torch.nn as nn
import torch.optim as optim
import scipy.signal as signal
import progressbar

#-------------------------------------------------------------------
# Class:    FxLMS algorithm
#-------------------------------------------------------------------
class FxLMS_AG_algorithm():
    
    def __init__(self, Len, Sec):
        '''
        Parameters:
        param1 - Len is the length of the control filter
        param2 - Sec is the secondary path [1 x L]
        '''
        self.Wc  = torch.zeros(1, Len, requires_grad=True, dtype=torch.float)
        self.Xd  = torch.zeros(1, Len, dtype=torch.float)
        self.Xm  = torch.zeros(Sec.shape[1], Len, dtype=torch.float)
        self.Yd  = torch.zeros(1, Sec.shape[1], dtype=torch.float)
        self.Sec = Sec
        self.Ls  = Sec.shape[1] 
        
    def feedforward(self, xin):
        self.Xd      = torch.roll(self.Xd,1,1)
        self.Xd[0,0] = xin 
        self.Xm      = torch.roll(self.Xm,1,0)
        self.Xm[0,:] = self.Xd 
        y            = self.Wc @ self.Xd.t()
        self.Yd      = torch.roll(self.Yd,1,1)
        self.Yd[0,0] = y.detach() 
        yt           = self.Yd @ self.Sec.t()
        Y            = self.Wc @ self.Xm.t()
        return yt, Y 
    
    def LossFunction(self,yt,d,Y):
        e          = d-yt 
        yt_e       = Y @ self.Sec.t()
        E_estimate = d - yt_e
        E_tr       = E_estimate.detach()
        Distance_est = E_estimate -e 
        return 2*e*E_estimate,e,Distance_est
    
    def _get_coeff_(self):
        return self.Wc.detach().numpy()

#------------------------------------------------------------------------------
# Function : train_fxlms_algorithm() 0.00000005
#------------------------------------------------------------------------------
def train_fxlms_algorithm(Model, Ref, Disturbance, Stepsize = 0.00001):
   
    bar = progressbar.ProgressBar(maxval=2*Disturbance.shape[0], \
        widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    
    #Stepsize = 0.00000005  
    optimizer= optim.SGD([Model.Wc], lr=Stepsize)
    
    bar.start()
    Erro_signal = []
    Dist_record = []
    len_data = Disturbance.shape[0]
    for itera in range(len_data):
        # Feedfoward
        xin = Ref[itera]
        dis = Disturbance[itera]
        yt,Y    = Model.feedforward(xin)
        loss,e, Distance_est = Model.LossFunction(yt,dis,Y)
        
            # Progress shown 
        bar.update(2*itera+1)
            
        # Backward 
        optimizer.zero_grad() 
        loss.backward()
        optimizer.step()
        Erro_signal.append(e.item())
        Dist_record.append(Distance_est.item())
        
        # Progress shown 
        bar.update(2*itera+2)
    bar.finish()
    return Erro_signal, Dist_record

#------------------------------------------------------------
# Function : Generating the testing bordband noise 
#------------------------------------------------------------
def Generating_boardband_noise_wavefrom_tensor(Wc_F, Seconds, fs):
    filter_len = 1024 
    bandpass_filter = signal.firwin(filter_len, Wc_F, pass_zero='bandpass', window ='hamming',fs=fs) 
    N = filter_len + Seconds*fs
    xin = np.random.randn(N)
    y   = signal.lfilter(bandpass_filter,1,xin)
    yout= y[filter_len:]
    # Standarlize 
    yout = yout/np.sqrt(np.var(yout))
    # return a tensor of [1 x sample rate]
    return torch.from_numpy(yout).type(torch.float).unsqueeze(0)

#-------------------------------------------------------------
# Function    : Disturbance_reference_generation_from_Fvector()
# Discription : Generating the distubrane and reference signal from the defined parameters
#-------------------------------------------------------------
def Disturbance_Noise_generation_from_Fvector(fs, T, f_vector, Pri_path, Sec_path):
    """
    Pri_path and Sec_path are  One dimension arraies 
    """
    # ANC platform configuration
    t     = np.arange(0,T,1/fs).reshape(-1,1)
    len_f = 1024
    b2    = signal.firwin(len_f, [f_vector[0],f_vector[1]], pass_zero='bandpass', window ='hamming',fs=fs)
    xin   = np.random.randn(len(t))
    Re    = signal.lfilter(b2,1,xin)
    Noise = Re[len_f-1:]
    # Noise = Noise/np.sqrt(np.var(Noise))
    
    # Construting the desired signal 
    Dir, Fx = signal.lfilter(Pri_path, 1, Noise), signal.lfilter(Sec_path, 1, Noise)
    
    return torch.from_numpy(Dir).type(torch.float), torch.from_numpy(Noise).type(torch.float)

#-------------------------------------------------------------------
if __name__ == '__main__':
    # Len = 25 
    # Wc = torch.zeros(1, Len, requires_grad=True, dtype=torch.float)
    # Xd = torch.zeros(1, Len, dtype= torch.float)
    # print(f'Wc size is {Wc.shape}' + ' && ' + f'Xd size is {Xd.shape}')
    # A = torch.einsum('ij,nj->in', Wc, Xd).squeeze(0)
    # print(f'A size is {A.shape}' + '&&' + f'value is {A}')
    # D = Wc.detach()
    # print(D)
    # C = Wc @ Xd.t()
    # print(C.shape)
    # Xc = torch.zeros(3,4,dtype=torch.float)
    # print(f'The tsting size is {Xc.shape}')
    # Xt = Xc.unsqueeze(1)
    # print(f'The second tsting size is {Xt.shape}')
    
    # Wc = torch.zeros(3, 2, Len, requires_grad=True, dtype=torch.float)
    # Xd = torch.zeros(3, 1, Len, dtype=torch.float)
    # y_mac        = torch.einsum('rtl,rtl->rt',Xd,Wc)
    # print(f'The third tsting size is {y_mac.shape}')
    # Ts = torch.tensor(
    #     [ [1, 3, 4],
    #       [4, 5, 6]
    #     ],
    # dtype=torch.float)
    # print(Ts.shape)
    # C = torch.einsum('rs->r',Ts)
    # print(C)
    # #-----------------------------------------------------
    # Ts1 = torch.tensor(
    #     [[1, 3, 4],
    #      [7, 8, 9], 
    #      [1, 1, 1]],
    # dtype= torch.float)
    
    # Tc1 = torch.tensor(
    #     [2, 3, 1],
    # dtype= torch.float)
    # D = torch.einsum('mn, mn->m',Ts1, Tc1.unsqueeze(0))
    # print(Ts1[:,0].shape)
    x = torch.tensor(
       [ 
        [1, 3, 5],
        [2, 4, 8],
        [3, 6, 7]
       ])
    print(f'R_num = {x.shape[0]} x Ls ={x.shape[1]}')
    
    y = x.repeat(2, 1, 1)
    print(y.shape)
    S = torch.permute(y,(1,0,2))
    print(S.shape)
    print(S)
    #print(x.unsqueeze(1).unsqueeze(1).shape)
    print(1/torch.einsum('RSE,RSE->RS',S,S))
    pass