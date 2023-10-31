import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn import svm
from sklearn.neural_network import MLPRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import pandas as pd
import matplotlib.pyplot as plt
from preprocessing import normalize
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

def evaluate_withmodels(x,y,names, n):
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
        listy = []
        while i < 30:
            x_train, x_test, y_train, y_test = new_trainingandtestsplit(x, y, names, n)
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            y_pred = np.maximum(y_pred, 0)
            # mae = mean_absolute_percentage_error(list(y_test['conc_GSH'].replace(0, 1)), y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            listy.append(mae)
            i += 1
        # joblib.dump(model,'models/GBRT_cut_data_wconc_allsol_580_BR_NM_10com.pkl')
        results.append([j,'30 fold cv score:', np.average(listy)])
        # print(y_pred, y_test)
        # plt.scatter(y_pred, y_test)
        # plt.plot(np.arange(0, 90), np.arange(0, 90))
        # plt.show()
    print(results)
    return None


def new_trainingandtestsplit(x, y, names, split):
    def categorize_row(row):
        if 'BSA' in row:
            return 1
        elif 'PEG' in row:
            return 2
        else:
            return 3
    names = names.applymap(categorize_row)
    ones = (names == 1).sum() * split
    twos = (names == 2).sum() * split
    threes = (names == 3).sum() * split
    splits = [int(ones), int(twos), int(threes)]
    names['keys'] = [i for i in range(len(names))]
    names = names.sort_values(by='names')
    testingx = []
    testingy = []
    trainingx = []
    trainingy = []
    for i in range(1,3):
        n = splits[i-1]
        filtered_names = names[names['names']==int(i)]
        head_names = filtered_names.head(n)
        remaining_names = filtered_names.iloc[n:]
        for i in head_names['keys']:
            trainingx.append(x.iloc[i])
            trainingy.append(y.iloc[i])

        for i in remaining_names['keys']:
            testingx.append(x.iloc[i])
            testingy.append(y.iloc[i])

    testingx = pd.concat(testingx, ignore_index=True, axis=1).T
    testingy = pd.concat(testingy, ignore_index=True, axis=1).T

    trainingx = pd.concat(trainingx, ignore_index=True, axis=1).T
    trainingy = pd.concat(trainingy, ignore_index=True, axis=1).T

    return trainingx, testingx, trainingy, testingy


if __name__ == '__main__':
    x1 = pd.read_csv('data/test.csv')
    y1 = pd.read_csv('data/data_580_concentrations_GSSG.csv')
    names = pd.read_csv('data/data_580_names.csv')


    evaluate_withmodels(x1, y1, names, .75)


    # how to find what spectra are better predicted?
    # link names and test vals



