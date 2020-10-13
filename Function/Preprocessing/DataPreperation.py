import numpy as np
import joblib
import sys
from Data import DataReader as DR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
class Data_Preperation:
    def __init__(self, mode,*,data_set,DP=None):
        self.input_set=data_set[0]
        self.target_set=data_set[1]
        self.Label={}
        self.InverseLabel = {"X_train": 0, "y_train": 1, "X_test": 2, "y_test": 3,"X":0,"y":1}
        self.Merge_Dataset=[]
        self.Scaled_Dataset=[]
        self.input_feature=self.input_set[0].shape[1]
        self.target_feature=self.target_set[0].shape[1]
        self.scaler=StandardScaler()

        if mode=="Training":
            self.Split_Dataset = []
            self.Merge_Split_Dataset = []
            self.scaler.fit(self.MergeData()[0])

        if mode=="Predicting":
            self.scaler=DP.scaler

    def AddParameter(self,L,number):
        for i in range(number):
            L.append({})

    def DataSplit(self,test_size=0.2,del_size=0.2,random_state=233):
        """
        Input- and Target Data Set are split up into Training, Development and Test Data Set.
        :param test_size: [float], Size of Test Data Set.
        :param del_size: [float], Size of Development Data Set.
        :param random_state: [float], Random Seed.
        :return:
        Split_Dataset[X_train, y_train, X_test, y_test, X_del, y_del].
        Split_Dataset[0] X_train: [Dict]. Key is the number of the Data Set, value is Training Data Input.
        Split_Dataset[1] y_train: [Dict]. Key is the number of the Data Set, value is Training Data Target.
        Split_Dataset[2] X_test: [Dict]. Key is the number of the Data Set, value is Test Data Input.
        Split_Dataset[3] y_test: [Dict]. Key is the number of the Data Set, value is Test Data Target.
        (Split_Dataset[4] X_del: [Dict]. Key is the number of the Data Set, value is Development Data Input.
        Split_Dataset[5] y_del: [Dict]. Key is the number of the Data Set, value is Development Data Target.)
        """
        # X_train, y_train, X_test, y_test, X_del, y_del
        Split_Dataset=[{}, {}, {}, {}]
        self.Label={0:"X_train",1:"y_train",2:"X_test",3:"y_test"}
        if not del_size==0:
            self.AddParameter(Split_Dataset,2)
            self.Label[4]="X_del"
            self.Label[5]="y_del"
            self.InverseLabel["X_del"]=4
            self.InverseLabel["y_del"]=5

        for i in range(len(self.input_set)):
            Split_Dataset[0][i], Split_Dataset[2][i], Split_Dataset[1][i], Split_Dataset[3][i]= \
                train_test_split(self.input_set[i],self.target_set[i],test_size=test_size,random_state=random_state)

            if not del_size==0:
                Split_Dataset[0][i], Split_Dataset[4][i], Split_Dataset[1][i], Split_Dataset[5][i]= \
                    train_test_split(Split_Dataset[0][i], Split_Dataset[1][i], test_size=del_size/(1-test_size), random_state=random_state)

        self.Split_Dataset=Split_Dataset
        return Split_Dataset

    def MergeData(self):
        """
        Merge the Data set.
        :return:
        Merge_Dataset: [List], [X, y].
        Merge_Dataset[0]: [ndarray], Merged Input set.
        Merge_Dataset[1]: [ndarray],Merged Target set
        """
        self.Label={0:"X",1:"y"}
        Merge_Dataset=[self.input_set[0],self.target_set[0]]
        for i in range(1,len(self.input_set)):
            Merge_Dataset[0]=np.append(Merge_Dataset[0],self.input_set[i],axis=0)
            Merge_Dataset[1]=np.append(Merge_Dataset[1],self.target_set[i],axis=0)
        print(type(Merge_Dataset))
        self.Merge_Dataset=Merge_Dataset
        return Merge_Dataset

    def MergeSplitData(self):
        """
        Merge the already split Dataset(for Training)
        :return:
        Merge_Split_Dataset: [List], [X_train, y_train, X_test, y_test, X_del, y_del].
        Merge_Split_Dataset[0]: [ndarray], merged Training Input Set
        Merge_Split_Dataset[1]: [ndarray], merged Training Target Set
        Merge_Split_Dataset[2]: [ndarray], merged Test Input Set
        Merge_Split_Dataset[3]: [ndarray], merged Test Target Set
        (Merge_Split_Dataset[4]: [ndarray], merged Development Input Set
        Merge_Split_Dataset[5]: [ndarray], merged Development Target Set)
        """
        Merge_Split_Dataset=[]
        for i in range(len(self.Split_Dataset)):
            Merge_Split_Dataset.append(self.Split_Dataset[i][0])
            if not len(self.Split_Dataset[0])==1:
                for j in range(len(self.Split_Dataset[i])):
                    if j>0:
                        Merge_Split_Dataset[i]=np.append(Merge_Split_Dataset[i], self.Split_Dataset[i][j],axis=0)
        self.Merge_Split_Dataset=Merge_Split_Dataset
        return Merge_Split_Dataset

    def DataScaling(self,Merge_Dataset,Load_Model=False,Unify=True,Path=None,Mean=False,Var=False):

        """
        Use Mean Value and Standard Deviation of features to scale the Merged Data Set(for Training / Predicting).
        :param Merge_Dataset: [List], For Training: Merge_Split_Dataset, [X_train, y_train, X_test, y_test, X_del, y_del]
                                      For Predicting: Merge_Dataset, [input, target]
        :param Load_Model: [boolean], When True, Input the existed scaler from external .joblib file.
        :param Unify: [boolean], Use unified Mean and Std.(True) or seperate Mean and Std.(False) to scale the Dataset.
        :param Path: [Str], Path of the existed scaler.
        :param Mean: [boolean], show Mean value of each feature when "True".
        :param Var:  [boolean], shown Variance of each feature when "True".
        :return:
        Scaled_Dataset for Training: [X_train, y_train, X_test, y_test, X_del, y_del]
                       for Predicting: [X, y]
        Scaled_Dataset[0]: [ndarray], Scaled Training Input Set/input Set
        Scaled_Dataset[1]: [ndarray], Scaled Training Target Set/target Set
        Scaled_Dataset[2]: [ndarray], Scaled Test Input Set
        Scaled_Dataset[3]: [ndarray], Scaled Test Target Set
        (Scaled_Dataset[4]: [ndarray], Scaled Development Input Set
        Scaled_Dataset[5]: [ndarray], Scaled Development Target Set)
        """
        Scaled_Dataset = []

        # Perform Scaling
        if Load_Model==False:
            scaler=StandardScaler()
            if Unify==False:
                print("Use different Scaler!")
            else:
                print("Use Unified Scaler!")
                if Mean==True:
                    print(f"Mean value of Unified Data:{self.scaler.mean_}")
                if Var==True:
                    print(f"Variation of Unified Data:{self.scaler.var_}")

            for i in range(len(Merge_Dataset)):
                Scaled_Dataset.append("")

                # Scale the input feature
                if i%2==0:

                    # Only Scale the input feature with different Mean and Std
                    if Unify==False:
                        Scaled_Dataset[i]=scaler.fit_transform(Merge_Dataset[i])
                        if Mean == True:
                            print(f"Mean value of {self.Label[i]}: {scaler.mean_}")
                        if Var == True:
                            print(f"Variation of {self.Label[i]}. Set is {scaler.var_}")

                    # Scale the input feature with unified Mean and Std
                    else:
                        Scaled_Dataset[i] = self.scaler.transform(Merge_Dataset[i])

                else:
                    Scaled_Dataset[i]=Merge_Dataset[i]

        # Scale with existed Scaler
        else:
            scaler=joblib.load(Path)
            for i in range(len(Merge_Dataset)):
                Scaled_Dataset.append("")
                if i%2==0:
                    Scaled_Dataset[i]=scaler.transform(Merge_Dataset[i])
                else:
                    Scaled_Dataset[i]=Merge_Dataset[i]
            print("Use existed scaler, mean value:\n",scaler.mean_)

        self.Scaled_Dataset=Scaled_Dataset

        return Scaled_Dataset



# Test
# path = os.path.abspath(os.path.join(os.getcwd(), "../.."))
# print(sys.path)
# dr=DR.DataReader("../../Data/Training Data","daten1P","daten2P","daten3P")
# dp=Data_Preperation(*dr.LoadData())
# print(f"Input_feature:{dp.input_feature},Target_feature:{dp.target_feature}")
# print("Dimension of 1.Input Data Set:",dp.input_set[0].shape)
# print("Dimension of 2.Input Data Set:",dp.input_set[1].shape)
# d=dp.DataSplit(0.2,0.2)
# print("Dimension of Split Data Set:",len(dp.MergeSplitData(d)))
# print("Dimension of Merge X_train:",(dp.MergeSplitData(d))[0].shape)
# print("Dimension of 1. X_train:",d[0][0].shape)
# print("Dimension of 2. X_train:",d[0][1].shape)
# print("Dimension of 3. X_train:",d[0][2].shape)
# print("Dimension of 1. X_test:",d[2][0].shape)
# print("Dimension of 2. X_test:",d[2][1].shape)
# print("Dimension of 3. X_test:",d[2][2].shape)
# print(Scaled_d[0])
# print(d[5][0].shape)
# print(d[5][1].shape)
# print(len((dp.input_set)))