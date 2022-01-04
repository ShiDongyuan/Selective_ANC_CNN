import  numpy as np 

def NR_level_compute(Disturbance, Error):
    Power_dis = 10*np.log10(np.var(Disturbance))
    Power_err = 10*np.log10(np.var(Error))
    Nr_level  = Power_dis - Power_err
    return Nr_level

if __name__ == "__main__":
    Re = np.random.rand(32)
    Di = np.random.rand(32)
    
    Nr = NR_level_compute(Di,Re)
    print(f'the noise reduction level is {Nr} dB !!')
    
    pass 