import numpy as np
import sys
from Data import DataReader as DR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
class DataPreperation:
    def __init__(self, *data_set):
        self.input_set=data_set[0]
        self.target_set=data_set[1]

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
        Split_Dataset(X_train, y_train, X_test, y_test, X_del, y_del).
        Split_Dataset[0] X_train: [Dict]. Key is the number of the Data Set, value is Training Data Input.
        Split_Dataset[1] y_train: [Dict]. Key is the number of the Data Set, value is Training Data Target.
        Split_Dataset[2] X_test: [Dict]. Key is the number of the Data Set, value is Test Data Input.
        Split_Dataset[3] y_test: [Dict]. Key is the number of the Data Set, value is Test Data Target.
        (Split_Dataset[4] X_del: [Dict]. Key is the number of the Data Set, value is Development Data Input.
        Split_Dataset[5] y_del: [Dict]. Key is the number of the Data Set, value is Development Data Target.)
        """
        # X_train, y_train, X_test, y_test, X_del, y_del
        Split_Dataset=[{}, {}, {}, {}]
        if not del_size==0:
            self.AddParameter(Split_Dataset,2)

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

# Test
# path = os.path.abspath(os.path.join(os.getcwd(), "../.."))
# print(sys.path)
dr=DR.DataReader("../../Data/Training Data","daten1P","daten2P","daten3P")
dp=DataPreperation(*dr.LoadData())
print("Dimension of 1.Input Data Set:",dp.input_set[0].shape)
print("Dimension of 2.Input Data Set:",dp.input_set[1].shape)
d=dp.DataSplit(0.2,0.2)
print("Dimension of Split Data Set:",len(dp.MergeSplitData(d)))
print("Dimension of Merge X_train:",(dp.MergeSplitData(d))[0].shape)
print("Dimension of 1. X_train:",d[0][0].shape)
print("Dimension of 2. X_train:",d[0][1].shape)
print("Dimension of 3. X_train:",d[0][2].shape)
print("Dimension of 1. X_test:",d[2][0].shape)
print("Dimension of 2. X_test:",d[2][1].shape)
print("Dimension of 3. X_test:",d[2][2].shape)
# print(d[5][0].shape)
# print(d[5][1].shape)
# print(len((dp.input_set)))