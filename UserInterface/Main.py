from Data import DataReader as DR
from Function.Preprocessing import DataPreperation as DP
from Function.Training import Optimization as Op
from Function.Training import Training as Tr
from Function.Predicting import Prediction

def main():
    """
    User's Interface.
    :return:
    void
    """
    # Pre Processing
    file=[]
    fileC=[]
    print("SHM Demonstrator is ready. Start Training!")
    while True:
        file_name=input("Please enter the name of Training Data Set:")
        file.append(file_name)
        if input("Last Data Set?")=="Y":
            break
    dr = DR.DataReader(f"../Data/Training Data", *file)
    dp = DP.Data_Preperation(*dr.LoadData())
    test_size=float(input("Size of Test Data?"))
    del_size=float(input("Size of Development Data?"))
    d=dp.DataSplit(test_size=test_size,del_size=del_size)
    Md=dp.MergeSplitData()
    dp.DataScaling(Md)

    # Processing
    h=Op.Hyperparameter()
    deep=int(input("Layers of MLP?"))
    print("Start Optimization...")
    h.HyperSearh(dp,deep=deep,iter=5,cv=3)

    # MLP Regressor using Optimized Hyperparameter
    pre=Tr.Regressor(DP=dp,op=h)

    # Evaluation
    Prediction.Predicting(pre.OptRegressor, dp)
    while True:
        Channel=input("Which System State?(Channel)")
        if Channel == "End":
            break
        else:
            Prediction.Predicting(pre.OptRegressor,dp,Channel=Channel)

    # Predicting
    if input("Start Predicting?")=="Y":
        while True:
            file_nameC = input("Please enter the name of Customer Data Set:")
            fileC.append(file_nameC)
            if input("Last Data Set?") == "Y":
                break
        dr1 = DR.DataReader(f"../Data/Customer Data", *fileC)
        d1 = dr1.LoadData(mode="Predicting")
        dp1 = DP.Data_Preperation(*d1)
        dp1.DataScaling(dp1.MergeData())
        Prediction.Predicting(pre.OptRegressor,dp1,Time_Domain=True)
        while True:
            Channel = input("Which System State?(Channel)")
            if Channel == "End":
                break
            else:
                Prediction.Predicting(pre.OptRegressor, dp1, Time_Domain=True, Channel=Channel)



# dr=DR.DataReader("../../Data/Training Data","daten1P","daten2P","daten3P")
# dp=DP(*dr.LoadData())
# d=dp.DataSplit(0.2,0.2)
# Md=dp.MergeSplitData(d)
# dp.DataScaling(Md)
# h=Op.Hyperparameter()
# h.HyperSearh(dp,deep=1,iter=5,cv=3)
# pre=Regressor.OptRegression(dp,h)
# Prediction.Predicting(pre,dp)