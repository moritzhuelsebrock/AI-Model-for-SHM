import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Data import DataReader as DR
from Function.Preprocessing.DataPreperation import Data_Preperation as DP
from sklearn.metrics import mean_absolute_error
from sklearn.neural_network import MLPRegressor

def Predicting(Regressor,DP,Time_Domain=False,Channel="Overview"):
    """
    Visualize the prediction.
    :param Regressor: [MLP Regressor], Initial or Optimized MLP Regressor.
    :param DP: [Data_Preperation], Data Set after Pre Processing.
    :param Time_Domain: [boolean], True: Predicting in Time Domain. False: Predicting the Training Result.
    :param Channel: [str], Expected Output System State.
    :return:
    void
    """
    #Use Project Data
    if Time_Domain==False:
        Prediction=Regressor.predict(DP.Scaled_Dataset[2])
        Target = DP.Scaled_Dataset[3]
    # Use Customer Data
    else:
        Prediction=Regressor.predict(DP.Scaled_Dataset[0])
        Target = DP.Scaled_Dataset[1]
    m2_T=Target[:,0]
    m3_T=Target[:,1]
    m4_T=Target[:,2]
    k_T=Target[:,3]
    alpha_T=Target[:,4]
    beta_T=Target[:,5]
    Time_Domain=Prediction.shape[0]
    m2=Prediction[:,0]
    m3=Prediction[:,1]
    m4=Prediction[:,2]
    k=Prediction[:,3]
    alpha=Prediction[:,4]
    beta=Prediction[:,5]
    if Channel=="Overview":
        plt.subplot(321)
        plt.scatter(range(Time_Domain),m2,marker="x",color="blue")
        plt.scatter(range(Time_Domain),m2_T,marker="o",color="red",alpha=0.6)
        plt.title("m2")
        plt.subplot(322)
        plt.scatter(range(Time_Domain),m3,marker="x",color="blue")
        plt.scatter(range(Time_Domain),m3_T,marker="o",color="red",alpha=0.6)
        plt.title("m3")
        plt.subplot(323)
        plt.scatter(range(Time_Domain), m4, marker="x", color="blue")
        plt.scatter(range(Time_Domain), m4_T, marker="o", color="red",alpha=0.6)
        plt.title("m4")
        plt.subplot(324)
        plt.scatter(range(Time_Domain), k, marker="x", color="blue")
        plt.scatter(range(Time_Domain), k_T, marker="o", color="red",alpha=0.6)
        plt.title("k")
        plt.subplot(325)
        plt.scatter(range(Time_Domain),alpha,marker="x",color="blue")
        plt.scatter(range(Time_Domain), alpha_T, marker="o", color="red",alpha=0.6)
        plt.title("alpha")
        plt.subplot(326)
        plt.scatter(range(Time_Domain),beta,marker="x",color="blue")
        plt.scatter(range(Time_Domain), beta_T, marker="o", color="red",alpha=0.6)
        plt.title("beta")
        plt.show()

    else:
        if Channel=="m2":
            plt.scatter(range(Time_Domain),m2,label="Prediction",marker="x",color="blue")
            plt.scatter(range(Time_Domain),m2_T,label="Target",marker="o",color="red",alpha=0.6)


        if Channel=="m3":
            plt.scatter(range(Time_Domain), m4, label="Prediction", marker="x", color="blue")
            plt.scatter(range(Time_Domain), m4_T, label="Target", marker="o", color="red",alpha=0.6)


        if Channel=="m4":
            plt.scatter(range(Time_Domain), m4, label="Prediction",marker="x", color="blue")
            plt.scatter(range(Time_Domain), m4_T, label="Target",marker="o", color="red", alpha=0.6)


        if Channel=="k":
            plt.scatter(range(Time_Domain), k, label="Prediction",marker="x", color="blue")
            plt.scatter(range(Time_Domain), k_T, label="Target",marker="o", color="red", alpha=0.6)


        if Channel=="alpha":
            plt.scatter(range(Time_Domain), alpha, label="Prediction",marker="x", color="blue")
            plt.scatter(range(Time_Domain), alpha_T, label="Target",marker="o", color="red", alpha=0.6)


        if Channel=="beta":
            plt.scatter(range(Time_Domain), beta, marker="x", label="Prediction",color="blue")
            plt.scatter(range(Time_Domain), beta_T, marker="o", label="Target",color="red", alpha=0.6)


        plt.title(Channel)
        plt.legend()
        mae = mean_absolute_error(m2_T, m2)
        print("Mean Absolute Error:", mae)
        print("Normalized Mean Absolute Error in %:", 100 * mae / np.ptp(m2_T))
        plt.show()







