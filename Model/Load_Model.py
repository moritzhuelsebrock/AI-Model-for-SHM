import joblib


def load_Preceptron(path, deep=3):
    """
    load the estimator from path
    print save path and type of estimator
    :param target_set: [narray],  Target data set, use to determine estimator class
    :param path: [str],  saving path
    :param deep: [int], depth of MLP, help to find joblib file
    :return estimator: [estimator],  MLP Perceptron model
    """
    # determine joblib file and save
    model_name = f"{path}/regressor_layer_{deep}.joblib"

    # Load from file
    estimator = joblib.load(model_name)

    return estimator

S=joblib.load("scaler.joblib")
print(S.mean_)