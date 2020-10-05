import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Data import DataReader as DR
from Function.Preprocessing.DataPreperation import Data_Preperation as DP
from sklearn.metrics import mean_absolute_error
from sklearn.neural_network import MLPRegressor

def Predicting(Regressor,DP,Time_Domain=False):
    #Use Project Data
    if Time_Domain==False:
        Prediction=Regressor.predict(DP.Scaled_Dataset[2])
    # Use Customer Data
    else:
        pass
    Target=DP.Scaled_Dataset[3]
    m2_T=Target[:,0]
    Time_Domain=Prediction.shape[0]
    m2=Prediction[:,0]
    m3=Prediction[:,1]
    m4=Prediction[:,2]
    k=Prediction[:,3]
    alpha=Prediction[:,4]
    beta=Prediction[:,5]
    plt.scatter(range(Time_Domain),m2,marker="x",color="blue")
    plt.scatter(range(Time_Domain),m2_T,marker="o",color="red")
    plt.show()



