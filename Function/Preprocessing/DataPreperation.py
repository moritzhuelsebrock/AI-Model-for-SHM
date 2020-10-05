import numpy as np
import sys
from Data import DataReader as DR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
class Data_Preperation:
    def __init__(self, *data_set):
        self.input_set=data_set[0]
        self.target_set=data_set[1]
        self.Label={}
        self.Scaled_Dataset=[]
        self.input_feature=self.input_set[0].shape[1]
        self.target_feature=self.target_set[0].shape[1]

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

        for i in range(len(self.input_set)):
            Split_Dataset[0][i], Split_Dataset[2][i], Split_Dataset[1][i], Split_Dataset[3][i]= \
                train_test_split(self.input_set[i],self.target_set[i],test_size=test_size,random_state=random_state)

            if not del_size==0:
                Split_Dataset[0][i], Split_Dataset[4][i], Split_Dataset[1][i], Split_Dataset[5][i]= \
                    train_test_split(Split_Dataset[0][i], Split_Dataset[1][i], test_size=del_size/(1-test_size), random_state=random_state)
        return Split_Dataset

    def MergeSplitData(self,Split_Dataset):
        """
        Merge the already split Data Set.
        :param Split_Dataset: [List], Dict of 6 Seperate Data Set
        :return:
        Merge_Dataset[X_train, y_train, X_test, y_test, X_del, y_del].
        Merge_Dataset[0]: [nparray], merged Training Input Set
        Merge_Dataset[1]: [nparray], merged Training Target Set
        Merge_Dataset[2]: [nparray], merged Test Input Set
        Merge_Dataset[3]: [nparray], merged Test Target Set
        (Merge_Dataset[4]: [nparray], merged Development Input Set
        Merge_Dataset[5]: [nparray], merged Development Target Set)
        """
        Merge_Dataset=[]
        for i in range(len(Split_Dataset)):
            Merge_Dataset.append(Split_Dataset[i][0])
            if not len(Split_Dataset[0])==1:
                for j in range(len(Split_Dataset[i])):
                    if j>0:
                        Merge_Dataset[i]=np.append(Merge_Dataset[i],Split_Dataset[i][j],axis=0)
        return Merge_Dataset

    def DataScaling(self,Merge_Dataset,Mean=False,Var=False):
        """
        Use Mean Value and Standard Deviation of features to scale the Merged Data Set.
        :param Merge_Dataset: [List], Merged Data Set [X_train, y_train, X_test, y_test, X_del, y_del].
        :param Mean: [boolean], show Mean value of each feature when "True".
        :param Var:  [boolean], shown Variance of each feature when "True".
        :return:
        Scaled_Dataset [X_train, y_train, X_test, y_test, X_del, y_del]
        Scaled_Dataset[0]: [nparray], Scaled Training Input Set
        Scaled_Dataset[1]: [nparray], Scaled Training Target Set
        Scaled_Dataset[2]: [nparray], Scaled Test Input Set
        Scaled_Dataset[3]: [nparray], Scaled Test Target Set
        (Scaled_Dataset[4]: [nparray], Scaled Development Input Set
        Scaled_Dataset[5]: [nparray], Scaled Development Target Set)
        """
        scaler=StandardScaler()
        Scaled_Dataset=[]
        for i in range(len(Merge_Dataset)):
            Scaled_Dataset.append("")
            Scaled_Dataset[i]=scaler.fit_transform(Merge_Dataset[i])
            if Mean==True:
                print(f"Mean value of {self.Label[i]}: {scaler.mean_}")
            if Var==True:
                print(f"Standard Deviation of {self.Label[i]}. Set is {scaler.var_}")
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