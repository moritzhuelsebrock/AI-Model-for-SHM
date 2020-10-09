import numpy as np
from Data import DataReader as DR
from Function.Preprocessing.DataPreperation import Data_Preperation as DP
from Function.Training import Optimization as Op
from sklearn.neural_network import MLPRegressor
from Function.Predicting import Prediction
class Regressor:
    def __init__(self,DP,op,Ini=False):
        if Ini==False:
            self.OptRegressor=self.OptRegression(DP,op)
        else:
            self.IniRegressor=self.IniRegression(DP)
    def OptRegression(self,DP,Op):
        """
        Training the optimal MLP Regressor.
        :param DP: [Data_Preperation], Dataset after Pre Processing.
        :param Op: [Hyperparameter], Optimized Hyperparameter.
        :return:
        OptRegressor: [MLP Regressor],Trained MLP Regressor with optimal Hyperparameter.
        """
        OptRegressor=MLPRegressor(hidden_layer_sizes=Op.HiddenLayerSize,alpha=Op.alpha,max_iter=Op.Max_iter*2)
        OptRegressor.fit(DP.Scaled_Dataset[DP.InverseLabel["X"]],DP.Scaled_Dataset[DP.InverseLabel["y"]])
        # OptRegressor.score(DP.Scaled_Dataset[2],DP.Scaled_Dataset[3])
        # Prediction=OptRegressor.predict(DP.Scaled_Dataset[2])
        return OptRegressor
        # return Prediction
    def IniRegression(self,DP):
        IniRegressor=MLPRegressor(solver="lbfgs", alpha=4.175e-05, hidden_layer_sizes=(53, 26, 25), max_iter=10000)
        IniRegressor.fit(DP.Scaled_Dataset[DP.InverseLabel["X"]],DP.Scaled_Dataset[DP.InverseLabel["y"]])
        return IniRegressor


# Test
# dr=DR.DataReader("../../Data/Training Data","daten1P","daten2P","daten3P")
# dp=DP(*dr.LoadData())
# d=dp.DataSplit(0.2,0.2)
# Md=dp.MergeSplitData(d)
# dp.DataScaling(Md)
# h=Op.Hyperparameter()
# h.HyperSearh(dp,deep=1,iter=5,cv=3)
# pre=Regressor.OptRegression(dp,h)
# Prediction.Predicting(pre,dp)