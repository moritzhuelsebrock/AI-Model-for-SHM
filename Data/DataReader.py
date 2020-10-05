import pandas as pd
import numpy as np
class DataReader(object):
    label_x = ['omega_1', 'omega_2', 'omega_3', 'D_1', 'D_2', 'D_3', 'EVnorm1_1', 'EVnorm1_2', 'EVnorm1_3',
               'EVnorm2_1', 'EVnorm2_2', 'EVnorm2_3', 'EVnorm3_1', 'EVnorm3_2', 'EVnorm3_3', 'Tem']
    label_y = ['m2', 'm3', 'm4', 'k', 'alpha', 'beta']

    def __init__(self,path='Training Data',*file_name,type="csv"):
        """
        Constructor of DataReader.
        :param path: Folder of the Data Set.
        :param file_name: Name of the Data Set.
        :param type: Type of the Data Set(Default: csv)
        """
        self.path=path
        self.file_name=file_name
        self.type=type

    def DataOverview(self,df):
        """
        Print the first 5 rows of Data Set and its Dimension for overview
        :param df:Data set, which will be overviewed
        :return:
        void
        """
        print(df.head(5))
        print("Dimension is",df.shape)

    def LoadData(self,Merge=False,overview='off'):
        """
        Load the Data Set. Loading Multiple Data sets are allowed.
        :param overview: [str], turn "on" to make an overview of Data Set.
        :return:
        Tuple(input_set,target_set)
        input_set: [Dict]. Key is the number of the Data Set, Value is ndarray, Labeled Input Data Set.
        target_set: [Dict]. Key is the number of the Data Set; Value is ndarray, Labeled Target Data Set.
        """
        df=[]
        input_set={}
        target_set={}
        count=0

        if Merge==False:

            for i in range(self.file_name.__len__()):
                df.append(pd.read_csv(f'{self.path}/{self.file_name[i]}.{self.type}'))
                input_set[i]=df[i][self.label_x].values
                target_set[i]=df[i][self.label_y].values

                if overview=='on':
                    self.DataOverview(df[i])
                    # print(df[i].head(5))
                    # print("Dimension is",df[i].shape)

                print(f"{i+1} Data set has been loaded successfully.")

        elif Merge==True:

            for i in range(self.file_name.__len__()):
                df.append(pd.read_csv(f'{self.path}/{self.file_name[i]}.{self.type}'))
                count = count + 1
            print(f"{count} Data Set has been concatenated successfully.")
            df_Concat = pd.concat(df)
            input_set[0] = df_Concat[self.label_x].values
            target_set[0] = df_Concat[self.label_y].values

        return input_set,target_set

    def LoadConcatData(self,overview="off"):
        """
         Load the Data Set and then concatenate them. Loading Multiple Data sets are allowed.
        :param overview: [str], turn "on" to make an overview of Data Set.
        :return:
        Tuple(input_set,target_set)
        Tuple[0] input_set: [ndarray], concatenated, labeled Input Data Set.
        Tuple[1] target_set: [ndarray]. concatenated, labeled Target Data Set.
        """
        df=[]
        count = 0
        for i in range(self.file_name.__len__()):
            df.append(pd.read_csv(f'{self.path}/{self.file_name[i]}.{self.type}'))
            count=count+1
        print(f"{count} Data Set has been concatenated successfully.")
        df_Concat=pd.concat(df)

        if(overview=="on"):
            self.DataOverview(df_Concat)
        input_set=df_Concat[self.label_x].values
        target_set=df_Concat[self.label_y].values

        return input_set,target_set

# Testing
# DR=DataReader("Training Data","daten1P")
# input_set,target_set=DR.LoadData()
# print(input_set[0].shape)
# input_set,target_set=DR.LoadConcatData()
# print(type(input_set))
# df=pd.read_csv("Training Data/daten1P.csv")