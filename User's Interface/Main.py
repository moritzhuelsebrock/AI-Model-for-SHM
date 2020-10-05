import numpy as np
from Data import DataReader as DR
from Function.Preprocessing import DataPreperation as DP
from Function.Training import Optimization as Op
from Function.Training import Training as Tr
from Function.Predicting import Prediction

def main():
    # Pre Processing
    file=[]
    Data_type=input("Training Data or Customer Data?")
    while True:
        file_name=input("Please enter the name of Data Set:")
        file.append(file_name)
        if input("Last Data Set?")=="Y":
            break
    dr=DR.DataReader(f"../Data/{Data_type}",*file)
    dp=DP.Data_Preperation(*dr.LoadData())
    test_size=float(input("Size of Test Data?"))
    del_size=float(input("Size of Development Data?"))
    d=dp.DataSplit(test_size=test_size,del_size=del_size)
    Md=dp.MergeSplitData(d)
    dp.DataScaling(Md)
    # Processing
    h=Op.Hyperparameter()
    deep=int(input("Layers of MLP?"))
    print("Start Optimization...")
    h.HyperSearh(dp,deep=deep,iter=5,cv=3)
    pre=Tr.Regressor(DP=dp,op=h)
    Prediction.Predicting(pre.Regression,dp)

main()

# dr=DR.DataReader("../../Data/Training Data","daten1P","daten2P","daten3P")
# dp=DP(*dr.LoadData())
# d=dp.DataSplit(0.2,0.2)
# Md=dp.MergeSplitData(d)
# dp.DataScaling(Md)
# h=Op.Hyperparameter()
# h.HyperSearh(dp,deep=1,iter=5,cv=3)
# pre=Regressor.OptRegression(dp,h)
# Prediction.Predicting(pre,dp)