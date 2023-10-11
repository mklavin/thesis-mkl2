import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn import svm
from sklearn.neural_network import MLPRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import pandas as pd
import math
import joblib
from sklearn.preprocessing import MinMaxScaler

def create_trainingandtest(x, y):
    indices = np.where(y.isna())[0]
    x = np.array(x)
    x = np.delete(x, indices, axis=0)
    y = y.dropna()
    y = np.array(y)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.15)
    return x_train, x_test, y_train, y_test


def evaluate_withmodels(x,y):
    # models:
    RF = RandomForestRegressor()
    SVM = svm.SVR()
    GBRT = GradientBoostingRegressor(alpha=.02, n_estimators=500)
    MLP = MLPRegressor(random_state=1, max_iter=500)
    KR = KernelRidge()
    KNN = KNeighborsRegressor()
    LR = LinearRegression()
    #HGBR = HistGradientBoostingRegressor(max_leaf_nodes=100)
    models = [KR, SVM, RF, GBRT, MLP, KNN, LR]

    results = []
    for j in models:
        model = j
        i = 0
        list = []
        while i < 30:
            x_train, x_test, y_train, y_test = create_trainingandtest(x, y)
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            mae = mean_absolute_error(y_test, y_pred)
            list.append(mae)
            i += 1
        # joblib.dump(model,'models/GBRT_cut_data_wconc_allsol_580_BR_NM_10com.pkl')
        results.append([j,'30 fold cv score:', np.average(list)])
        # print(y_pred, y_test)
    print(results)
    return None


if __name__ == '__main__':
    x1 = pd.read_csv('data/pca_data/allsol_610_BR_NM_20com.csv')
    x2 = pd.read_csv('data/prepro_methods/removerows_580_pre1_6com.csv')
    y1 = pd.read_csv('data/data_610_concentrations_GSH.csv')
    y2 = pd.read_csv('data/removerows_580_concentrations_GSSG.csv')


    evaluate_withmodels(x1,y1)

