from Data import DataReader as DR
from Function.Preprocessing import DataPreperation as DP
from Function.Training import Optimization as Op
from Function.Training import Training as Tr
from Function.Predicting import Prediction


files=["daten1P","daten2P","daten3P"]
files_R=["daten1_rauschen"]
files_T=["daten1_rauschen_sigma"]

# DataReader
dr_T=DR.DataReader("../Data/Training Data",*files)
dr_P=DR.DataReader("../Data/Customer Data",*files_T)

# Training
# DataPreperation
dp_T=DP.Data_Preperation(*dr_T.LoadData(mode="Training",overview="off"))
d=dp_T.DataSplit(0.2,0.2)
Md_T=dp_T.MergeSplitData()
dp_T.DataScaling(Md_T,Mean=False,Var=False)

# Optimization
h=Op.Hyperparameter()
h.HyperSearh(dp_T,deep=1, Development_Data=True, random_mode=True, iter=5, cv=3, OptInfo=False)
Model=Tr.Regressor(dp_T,h)
# Model=Tr.Regressor(dp_T,Ini=True)

# Prediction
Prediction.Predicting(Model.OptRegressor, dp_T, Time_Domain=False,Channel="Overview")
# Prediction.Predicting(Model.IniRegressor, dp_T, Time_Domain=False,Channel="Overview")

# Predicting
# DataPreperation
dp_P=DP.Data_Preperation(*dr_P.LoadData(mode="Predicting",overview="off"))
Md_P=dp_P.MergeData()
dp_P.DataScaling(Md_P,Mean=False,Var=False)

# Prdiction
Prediction.Predicting(Model.OptRegressor, dp_P, Time_Domain=True,  Channel="Overview")
Prediction.Predicting(Model.OptRegressor, dp_P, Time_Domain=True,  Channel="m2")
# Prediction.Predicting(Model.IniRegressor, dp_P, Time_Domain=True,  Channel="Overview")
# Prediction.Predicting(Model.IniRegressor, dp_P, Time_Domain=True,  Channel="m2")

