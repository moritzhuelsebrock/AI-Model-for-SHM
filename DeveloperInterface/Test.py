from Data import DataReader as DR
from Function.Preprocessing import DataPreperation as DP
from Function.Training import Optimization as Op
from Function.Training import Training as Tr
from Function.Predicting import Prediction
# from Model import Load_Model


files=["daten1P","daten2P","daten3P","daten4P","daten5P","daten6P","daten7P"]
files_R=["daten8P"]
files_T=["daten1_rauschen_sigma"]
files_T1=["daten2_rauschen_sigma"]

# DataReader
dr_T=DR.DataReader("../Data/Training Data",*files)
dr_P=DR.DataReader("../Data/Customer Data",*files_T)

# Training
# DataPreperation
Load_T=dr_T.LoadData(mode="Training",overview="off")
dp_T=DP.Data_Preperation("Training",data_set=Load_T)
d=dp_T.DataSplit(0.2,0.2)
Md_T=dp_T.MergeSplitData()
dp_T.DataScaling(Md_T,Mean=True,Var=True)

# Optimization
h=Op.Hyperparameter()
h.HyperSearh(dp_T,deep=1, Development_Data=True, random_mode=True, iter=5, cv=5, OptInfo=True)
Model=Tr.Regressor(dp_T,h)
# Model=Tr.Regressor(dp_T,h,Ini=True)

# # Load Model
# Model=Load_Model.load_Preceptron("../Model")


# Prediction
# Prediction.Predicting(Model, dp_T, Time_Domain=False,Channel="Overview")
Prediction.Predicting(Model.OptRegressor, dp_T, Time_Domain=False,Channel="Overview")
# Prediction.Predicting(Model.IniRegressor, dp_T, Time_Domain=False,Channel="Overview")

# Predicting
# DataPreperation
Load_P=dr_P.LoadData(mode="Predicting",overview="off")
dp_P=DP.Data_Preperation("Predicting",data_set=Load_P,DP=dp_T)
Md_P=dp_P.MergeData()
dp_P.DataScaling(Md_P,Mean=True,Var=True)

# Prdiction
# Prediction.Predicting(Model, dp_P, Time_Domain=True,  Channel="Overview")
# Prediction.Predicting(Model, dp_P, Time_Domain=True,  Channel="m2")
Prediction.Predicting(Model.OptRegressor, dp_P, Time_Domain=True,  Channel="Overview")
Prediction.Predicting(Model.OptRegressor, dp_P, Time_Domain=True,  Channel="m2")
Prediction.Predicting(Model.OptRegressor, dp_P, Time_Domain=True,  Channel="m3")
Prediction.Predicting(Model.OptRegressor, dp_P, Time_Domain=True,  Channel="m4")
Prediction.Predicting(Model.OptRegressor, dp_P, Time_Domain=True,  Channel="k")
Prediction.Predicting(Model.OptRegressor, dp_P, Time_Domain=True,  Channel="alpha")
Prediction.Predicting(Model.OptRegressor, dp_P, Time_Domain=True,  Channel="beta")
# Prediction.Predicting(Model.IniRegressor, dp_P, Time_Domain=True,  Channel="Overview")
# Prediction.Predicting(Model.IniRegressor, dp_P, Time_Domain=True,  Channel="m2")

