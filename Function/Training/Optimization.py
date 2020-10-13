import math
import time
import numpy as np
import pandas as pd
from Data import DataReader as DR
from Function.Preprocessing.DataPreperation import Data_Preperation as DP
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.neural_network import MLPRegressor
class Hyperparameter:
    def __init__(self):
        self.HiddenLayer=0
        self.HiddenLayerSize=[]
        self.alpha=0
        self.OptimizingTime=0
        self.Max_iter=0

    def SetOptHyperparameter(self,**Param):
        """
        Set Optimal Hyperparameter as Class attribute.
        :param Param: [dict], HiddenLayerSize, alpha, Max_iter, OptimizingTime...
        :return:
        void
        """
        self.HiddenLayerSize=Param["HiddenLayerSize"]
        self.alpha=Param["alpha"]
        self.Max_iter=Param["Max_iter"]
        self.OptimizingTime=Param["OptimizingTime"]


    def Initialization(self,deep):
        """
        Set up an initial MLP Regressor.
        :param deep: Numbers of Hidden Layer.
        :return:
        estimator: [MLP Regressor], inital MLP with initial parameter.
        """
        if deep == 1:
            estimator = MLPRegressor(solver="lbfgs", alpha=4.175e-05, hidden_layer_sizes=(53), max_iter=10000)

        if deep == 2:
            estimator = MLPRegressor(solver="lbfgs", alpha=4.175e-05, hidden_layer_sizes=(53, 26), max_iter=10000)

        if deep == 3:
            estimator = MLPRegressor(solver="lbfgs", alpha=4.175e-05, hidden_layer_sizes=(53, 26, 25), max_iter=10000)

        return estimator

    def HyperSearh(self,DP,deep=3, Development_Data=True, random_mode=True,iter=10,cv=5, OptInfo=False):
        """
        Optimization and then Assignment of Hyperparameter. Either Random Search or Grid Search.
        :param DP: [Data_Preperation], Dataset after Pre Processing.
        :param deep: [int], Number of Hidden Layer.
        :param Development_Data:[boolean], Using Development Data for optimization or not.
        :param random_mode: [boolean], True: Random Search, False: Grid Search.
        :param iter: [int], Number of Iteration Random Search.
        :param cv: [int] Cross Validation by K-Fold. cv is the Factor K.
        :param OptInfo: [boolean] Show concrete information of Optimization.
        :return:
        search_result: [Data Frame] Results of Optimization.
        """
        time_start=time.time()
        width = int(DP.input_feature*DP.target_feature/math.gcd(DP.input_feature,DP.target_feature))
        candidate_neuron = range(DP.target_feature, width)

        if deep==1:
            for layer_1 in candidate_neuron:
                self.HiddenLayerSize.append((layer_1))

        elif deep==2:
            for layer_2 in candidate_neuron:
                for layer_1 in candidate_neuron:
                    if layer_2<layer_1:
                        self.HiddenLayerSize.append((layer_1,layer_2))

        elif deep==3:
            for layer_3 in candidate_neuron:
                for layer_2 in candidate_neuron:
                    for layer_1 in candidate_neuron:
                        if layer_3 < layer_2 and layer_2 < layer_1:
                            self.HiddenLayerSize.append((layer_1, layer_2, layer_3))

        elif deep==4:
            if deep == 4:
                for layer_4 in candidate_neuron:
                    for layer_3 in candidate_neuron:
                        for layer_2 in candidate_neuron:
                            for layer_1 in candidate_neuron:
                                if layer_4 < layer_3 and layer_3 < layer_2 and layer_2 < layer_1:
                                    self.HiddenLayerSize.append((layer_1, layer_2, layer_3, layer_4))

        # Define the List of Hyperparmeter for Optimization
        param_space = {
        'hidden_layer_sizes': self.HiddenLayerSize,
        'alpha': np.logspace(-5, -2, 30),
        'max_iter': np.logspace(3, 4, 10)
            }

        estimator=self.Initialization(deep)

        if random_mode:
            hyper_search = RandomizedSearchCV(estimator, param_distributions=param_space,n_jobs=-1, n_iter=iter,cv=cv)
        else:
            hyper_search = GridSearchCV(estimator, param_grid=param_space,n_jobs=-1)


        if Development_Data==True:
            # Use Development Data to optimize
            hyper_search.fit(DP.Scaled_Dataset[DP.InverseLabel["X_del"]],DP.Scaled_Dataset[DP.InverseLabel["y_del"]])
        else:
            # Use Training Data/Customer Data to optimize
            hyper_search.fit(DP.Scaled_Dataset[DP.InverseLabel["X_train"]], DP.Scaled_Dataset[DP.InverseLabel["y_train"]])



        time_end = time.time()
        self.SetOptHyperparameter(HiddenLayerSize=hyper_search.best_params_["hidden_layer_sizes"],
                                  alpha=hyper_search.best_params_["alpha"],
                                  Max_iter=int(hyper_search.best_params_["max_iter"]),OptimizingTime=time_end-time_start)

        search_result=hyper_search.cv_results_

        if (OptInfo==True):
            self.report_search(search_result)
        else:
            self.report_search(search_result,Overview=False)

        return search_result


    def report_search(self,results, n_top=3, Overview=True):
        """
        Visualize the Result of optimization. Default: Show Top 3 Hyperparameter Set.
        :param results: [Data Frame], cv_Result from Hypersearch.
        :param n_top: [int], Top n Hyperparameter Set.
        :param Overview: [boolean], True: Concrete Information, False: Result of Optimization.
        :return:
        df: [Data Frame], cv_Result from Hypersearch.
        """
        df = pd.DataFrame(results)
        df = df.sort_values("rank_test_score")
        if Overview==True:
            for i in range(1,n_top+1):
                candidates = np.flatnonzero(results['rank_test_score'] == i)
                for candidate in candidates:
                    print(f"Model with rank: {i}")
                    print(f"Parameters: {results['params'][candidate]}")
                    print(f"Mean validation score: {results['mean_test_score'][candidate]:.3f}")
                    print("")
            print(f"Optimizing Time: {self.OptimizingTime}")
        else:
            candidates = np.flatnonzero(results['rank_test_score'] == 1)
            for candidate in candidates:
                print(f"Optimized Parameters: {results['params'][candidate]}")
            # para=df[df["rank_test_score"]==1]
            # print(para["params"])

        return df











# Test
# dr=DR.DataReader("../../Data/Training Data","daten1P","daten2P","daten3P")
# dp=DP(*dr.LoadData())
# d=dp.DataSplit(0.2,0.2)
# Md=dp.MergeSplitData(d)
# dp.DataScaling(Md)
# h=Hyperparameter()
# h.HyperSearh(dp,deep=1,iter=5,cv=3)