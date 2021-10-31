from Disturbance_generation import Disturbance_reference_generation
from FxLMS_algorithm import FxLMS_algroithm, train_fxlms_algorithm
import matplotlib.pyplot as plt 

if __name__ == "__main__":
    dis, fx = Disturbance_reference_generation() 
    controller = FxLMS_algroithm(Len=256)
    Erro = train_fxlms_algorithm(Model=controller,Ref=fx, Disturbance=dis)
    
    # Drawing the impulse response of the primary path
    plt.title('The response of the primary path')
    plt.plot(Erro)
    plt.ylabel('Amplitude')
    plt.xlabel('Time')
    plt.grid()
    plt.show()
    
    pass 