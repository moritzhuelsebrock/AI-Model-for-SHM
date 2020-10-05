import numpy as np
from Data import DataReader as DR
from Function.Preprocessing.DataPreperation import Data_Preperation as DP
from Function.Training import Optimization as Op
from sklearn.neural_network import MLPRegressor
from Function.Predicting import Prediction
class Regressor:
    def __init__(self,DP,op):
        self.Regression=self.OptRegression(DP,op)
    def OptRegression(self,DP,Op):
        OptRegressor=MLPRegressor(hidden_layer_sizes=Op.HiddenLayerSize,alpha=Op.alpha,max_iter=Op.Max_iter*2)
        OptRegressor.fit(DP.Scaled_Dataset[0],DP.Scaled_Dataset[1])
        OptRegressor.score(DP.Scaled_Dataset[2],DP.Scaled_Dataset[3])
        Prediction=OptRegressor.predict(DP.Scaled_Dataset[2])
        return OptRegressor
        # return Prediction
    def IniRegression(self):
        pass


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