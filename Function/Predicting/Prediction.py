import numpy as np
import pandas as pd
import statistics
import matplotlib.pyplot as plt
from Data import DataReader as DR
from Function.Preprocessing.DataPreperation import Data_Preperation as DP
from sklearn.metrics import mean_absolute_error
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

def InverseScaling(DP,Prediction,InverseLabel="y"):
    """
    Inverse Scale of the Target Dataset.
    :param DP: [Data_Preperation], Original Dataset
    :param Prediction: [ndarray], Target Dataset need to scale inversely.
    :param InverseLabel:[Str], Use its mean and std. to scale inversely.
    :return:
    InversePrediction: [ndarray], Inverse Scale of Target Dataset.
    """
    scaler=StandardScaler()

    # Use Mean, Var from y_test to scale inversely
    scaler.fit(DP.Merge_Dataset[DP.InverseLabel[InverseLabel]])
    InversePrediction= scaler.inverse_transform(Prediction)
    return InversePrediction

def Filter(Prediction, window=40, order=1):
    """
    Implementation of Mean Value Filter for Prediction.
    :param Prediction: [ndarray], Prediction need to be filtered.
    :param window: [int], Window size for Filter to compute the mean value.
    :param order: [int], Order of filter, implement the filter "order" times(Recursion).
    :return:
    Target_Result: [ndarray], Prediction after filtering.
    """
    shape=(Prediction.shape[0]-window,Prediction.shape[1])
    Filted_Result=np.empty(shape)
    for i in range(Prediction.shape[1]):
        for j in range(0,Prediction.shape[0]-window):
            Filted_Result[j][i]=statistics.mean(Prediction[j:j+window,i])

    Target_Result=Filted_Result

    # Recursion
    if order>1:
        Target_Result=Filter(Target_Result,window,order-1)

    return Target_Result

def Predicting(Regressor,DP,Time_Domain=False,Channel="Overview",FilterOrder=1):
    """
    Predicting and Visualize the prediction.
    :param Regressor: [MLP Regressor], Initial or Optimized MLP Regressor.
    :param DP: [Data_Preperation], Data Set after Pre Processing.
    :param Time_Domain: [boolean], True: Predicting in Time Domain. False: Predicting the Training Result.
    :param Channel: [str], Expected Output System State.
    :return:
    void
    """
    #Use Project Data
    if Time_Domain==False:
        Prediction = Regressor.predict(DP.Scaled_Dataset[DP.InverseLabel["X_test"]])
        # Prediction = InverseScaling(DP,Prediction)
        Target = DP.Scaled_Dataset[DP.InverseLabel["y_test"]]
        Assignment = AssignChannel(Target,Prediction)
        ShowMae(Assignment)
        Plot(Assignment,Channel)
    # Use Customer Data
    else:
        Prediction=Regressor.predict(DP.Scaled_Dataset[DP.InverseLabel["X"]])
        # Prediction = InverseScaling(DP, Prediction)
        Target = DP.Scaled_Dataset[DP.InverseLabel["y"]]
        Prediction_F = Filter(Prediction, order=FilterOrder)
        Assignment = AssignChannel(Target, Prediction_F)
        Plot(Assignment,Channel)
        return Prediction

def AssignChannel(Target,Prediction):
    """
    Assign the Output channel and extract the data from Prediction for analysing.
    :param Target: [ndarray], Value of Target Dataset.
    :param Prediction: [ndarray], Value of Prediction Dataset.
    :return:
    dict: [Dict], Dict of ndarray(Data Column), {m2, m3, m4, k, alpha, beta, Time_Domain,
    m2_T, m3_T, m4_T, k_T, alpha_T, beta_T, Time_Domain_F}
    """
    # Assign Output Channel(Target)
    dict = {}
    dict["m2_T"]=Target[:,0]
    dict["m3_T"]=Target[:,1]
    dict["m4_T"]=Target[:,2]
    dict["k_T"]=Target[:,3]
    dict["alpha_T"]=Target[:,4]
    dict["beta_T"]=Target[:,5]
    dict["Time_Domain"]=Target.shape[0]
    print(Prediction.shape)

    # Assign Output Channel(Prediction)
    dict["m2"]=Prediction[:,0]
    dict["m3"]=Prediction[:,1]
    dict["m4"]=Prediction[:,2]
    dict["k"]=Prediction[:,3]
    dict["alpha"]=Prediction[:,4]
    dict["beta"]=Prediction[:,5]
    dict["Time_Domain_F"]=Prediction.shape[0]

    return dict

    # Visualization
def ShowMae(Assignment):
    """
    Visualize the MAE and NAME of each channel.
    :param Assignment: [dict], Dictionary of Output Channel and Target Channel.
    :return:
    void
    """
    mae = mean_absolute_error(Assignment["m2_T"], Assignment["m2"])
    print("m2 Mean Absolute Error:", mae)
    print("m2 Normalized Mean Absolute Error in %:", 100 * mae / np.ptp(Assignment["m2_T"]))
    mae = mean_absolute_error(Assignment["m3_T"], Assignment["m3"])
    print("m3 Mean Absolute Error:", mae)
    print("m3 Normalized Mean Absolute Error in %:", 100 * mae / np.ptp(Assignment["m3_T"]))
    mae = mean_absolute_error(Assignment["m4_T"], Assignment["m4"])
    print("m4 Mean Absolute Error:", mae)
    print("m4 Normalized Mean Absolute Error in %:", 100 * mae / np.ptp(Assignment["m4_T"]))
    mae = mean_absolute_error(Assignment["k_T"],Assignment["k"])
    print("k Mean Absolute Error:", mae)
    print("k Normalized Mean Absolute Error in %:", 100 * mae / np.ptp(Assignment["k_T"]))
    mae = mean_absolute_error(Assignment["alpha_T"], Assignment["alpha"])
    print("alpha Mean Absolute Error:", mae)
    print("alpha Normalized Mean Absolute Error in %:", 100 * mae / np.ptp(Assignment["alpha_T"]))
    mae = mean_absolute_error(Assignment["beta_T"], Assignment["beta"])
    print("beta Mean Absolute Error:", mae)
    print("beta Normalized Mean Absolute Error in %:", 100 * mae / np.ptp(Assignment["beta_T"]))

def Plot(Assignment,Channel="Overview"):
    """
    Plot the scatter diagramm for training and predicting.
    :param Assignment: [dict], Dictionary of Output Channel and Target Channel.
    :param Channel: [Str], Overview or choose the specific channel to show details.
    :return:
    void.
    """
    if Channel=="Overview":
        plt.subplot(321)
        plt.scatter(range(Assignment["Time_Domain_F"]),Assignment["m2"],marker="x",color="blue")
        plt.scatter(range(Assignment["Time_Domain"]),Assignment["m2_T"],marker="o",color="red",alpha=0.6)
        plt.title("m2")
        plt.subplot(322)
        plt.scatter(range(Assignment["Time_Domain_F"]),Assignment["m3"],marker="x",color="blue")
        plt.scatter(range(Assignment["Time_Domain"]),Assignment["m3_T"],marker="o",color="red",alpha=0.6)
        plt.title("m3")
        plt.subplot(323)
        plt.scatter(range(Assignment["Time_Domain_F"]), Assignment["m4"], marker="x", color="blue")
        plt.scatter(range(Assignment["Time_Domain"]), Assignment["m4_T"], marker="o", color="red",alpha=0.6)
        plt.title("m4")
        plt.subplot(324)
        plt.scatter(range(Assignment["Time_Domain_F"]), Assignment["k"], marker="x", color="blue")
        plt.scatter(range(Assignment["Time_Domain"]), Assignment["k_T"], marker="o", color="red",alpha=0.6)
        plt.title("k")
        plt.subplot(325)
        plt.scatter(range(Assignment["Time_Domain_F"]),Assignment["alpha"],marker="x",color="blue")
        plt.scatter(range(Assignment["Time_Domain"]), Assignment["alpha_T"], marker="o", color="red",alpha=0.6)
        plt.title("alpha")
        plt.subplot(326)
        plt.scatter(range(Assignment["Time_Domain_F"]),Assignment["beta"],marker="x",color="blue")
        plt.scatter(range(Assignment["Time_Domain"]), Assignment["beta_T"], marker="o", color="red",alpha=0.6)
        plt.title("beta")
        plt.show()


    else:
        if Channel=="m2":
            plt.scatter(range(Assignment["Time_Domain_F"]),Assignment["m2"],label="Prediction",marker="x",color="blue")
            plt.scatter(range(Assignment["Time_Domain"]),Assignment["m2_T"],label="Target",marker="o",color="red",alpha=0.6)


        if Channel=="m3":
            plt.scatter(range(Assignment["Time_Domain_F"]), Assignment["m3"], label="Prediction", marker="x", color="blue")
            plt.scatter(range(Assignment["Time_Domain"]), Assignment["m3_T"], label="Target", marker="o", color="red",alpha=0.6)


        if Channel=="m4":
            plt.scatter(range(Assignment["Time_Domain_F"]), Assignment["m4"], label="Prediction",marker="x", color="blue")
            plt.scatter(range(Assignment["Time_Domain"]), Assignment["m4_T"], label="Target",marker="o", color="red", alpha=0.6)


        if Channel=="k":
            plt.scatter(range(Assignment["Time_Domain_F"]), Assignment["k"], label="Prediction",marker="x", color="blue")
            plt.scatter(range(Assignment["Time_Domain"]), Assignment["k_T"], label="Target",marker="o", color="red", alpha=0.6)


        if Channel=="alpha":
            plt.scatter(range(Assignment["Time_Domain_F"]), Assignment["alpha"], label="Prediction",marker="x", color="blue")
            plt.scatter(range(Assignment["Time_Domain"]), Assignment["alpha_T"], label="Target",marker="o", color="red", alpha=0.6)


        if Channel=="beta":
            plt.scatter(range(Assignment["Time_Domain_F"]), Assignment["beta"], marker="x", label="Prediction",color="blue")
            plt.scatter(range(Assignment["Time_Domain"]), Assignment["beta_T"], marker="o", label="Target",color="red", alpha=0.6)


        plt.title(Channel)
        plt.legend()
        # mae = mean_absolute_error(m2_T, m2)
        # print("Mean Absolute Error:", mae)
        # print("Normalized Mean Absolute Error in %:", 100 * mae / np.ptp(m2_T))
        plt.show()







