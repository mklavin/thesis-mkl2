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
from sklearn.model_selection import GridSearchCV

def find_data_onbadspec(listy, x, y, names):
    """
    :param listy: list of spectra that returned error specified by if statement
    :param x: original features
    :param y: original labels
    :param names: names of the spectra in x
    :return: print the indices, concentrations, and
    """
    x = np.array(x)
    y = np.array(y)
    indices = []
    conc = []
    namey = []
    for i in range(len(listy)):
        row_index = np.where(np.all(x == listy[i], axis=1))[0]
        indices.append(row_index)
        conc.append((y[row_index]))
        namey.append(names.iloc[row_index])
    print('indices:', indices)
    print('concentrations:', conc)
    print('names', namey)
    return None

def evaluate_withmodels(x, y, names, n):
    # x = x.drop([10])
    # y = y.drop([10])
    # names = names.drop([10])

    # models:
    RF = RandomForestRegressor()
    SVM = svm.SVR(kernel='poly')
    GBRT = GradientBoostingRegressor(alpha=.001, n_estimators=50000)
    MLP = MLPRegressor(random_state=1, hidden_layer_sizes=(100, ), solver='lbfgs', max_iter=5000)
    KR = KernelRidge()
    KNN = KNeighborsRegressor()
    LR = LinearRegression()
    #HGBR = HistGradientBoostingRegressor(max_leaf_nodes=100)
    models = [RF, SVM, GBRT, MLP, KR, KNN, LR]

    badspec = []
    results = []
    for j in models:
        model = j
        i = 0
        listy = []
        while i < 4:
            x_train, x_test, y_train, y_test = new_trainingandtestsplit(x, y, names, n)
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            y_pred = np.maximum(y_pred, 0)
            # mae = mean_absolute_percentage_error(list(y_test['conc_GSH'].replace(0, 1)), y_pred)

            for i in range(len(y_pred)):
                try:
                    if mean_absolute_error(y_test.iloc[i], y_pred[i]) > 1:
                        badspec.append(list(x_test.iloc[i]))
                except TypeError:
                    if mean_absolute_error(y_test.iloc[i], [y_pred[i]]) > 1:
                        badspec.append(list(x_test.iloc[i]))

            mae = mean_absolute_error(y_test, y_pred)
            listy.append(mae)
            i += 1
        # joblib.dump(model,'models/GBRT_cut_data_wconc_allsol_580_BR_NM_10com.pkl')
        results.append([j,'MAE:', np.average(listy)])
    print(results)
    return find_data_onbadspec(badspec, x, y, names) # can change this arg- right now it finds which spectra have the highest errors


def new_trainingandtestsplit(x, y, names, split):
    # makes a training and test split that has an even amount of PEG, BSA, and phos solvent samples
    # sorts them by names
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

def evaluate_fake_data_withmodels(x, y, fake_x, fake_y):
    # models:
    RF = RandomForestRegressor()
    SVM = svm.SVR()
    GBRT = GradientBoostingRegressor(alpha=.001, n_estimators=50000)
    MLP = MLPRegressor(random_state=1, hidden_layer_sizes=(100, ), solver='lbfgs', max_iter=5000)
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
        while i < 8:
            model.fit(fake_x, fake_y)
            y_pred = model.predict(x)
            y_pred = np.maximum(y_pred, 0)
            mae = mean_absolute_error(y, y_pred)
            listy.append(mae)
            i += 1
        # joblib.dump(model,'models/GBRT_cut_data_wconc_allsol_580_BR_NM_10com.pkl')
        results.append([j,'MAE:', np.average(listy)])
    print(results)
    return None

def tune_param(x_train, y_train):
    # Define the SVR model
    svr = svm.SVR()

    # Define the parameter grid to search
    param_grid = {
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'C': [0.1, 1, 10],
        'gamma': ['scale', 'auto']
    }

    # Create GridSearchCV
    grid_search = GridSearchCV(svr, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)

    # Fit the model to the training data
    grid_search.fit(x_train, y_train)

    # Print the best parameters and best score
    print("Best Parameters: ", grid_search.best_params_)
    print("Best Score: ", grid_search.best_score_)

    return grid_search.best_estimator_

if __name__ == '__main__':
    x1 = pd.read_csv('data/updated_12_25_data_prepro_610.csv')
    y1 = pd.read_csv('data/updated_12_25_data_610_concentrations_GSH.csv')
    names = pd.read_csv('data/updated_12_25_data_610_names.csv')

    evaluate_withmodels(x1, y1, names, .85)
    exit()

    df = pd.concat([x1, y1], axis=1)
    print(df)
    df = pd.DataFrame(np.corrcoef(df.T))
    print(df)
    sorted_df = df.sort_values(by=1339, ascending=False)
    print(sorted_df[1339])
    exit()

    y_values = np.array(y1)
    # Perform the division
    normalized_y = y_values / np.max(y_values)

    x_values = np.array(x1['150'])
    # Perform the division
    normalized_x = x_values / np.max(x_values)


    exit()

    evaluate_withmodels(x1, y1, names, .75)


    # MLP's LBFGS solver works best for 610 region and MLP's SGD solver works best for 580 region
    # SVM linear solver works best for 580 region

    # why is the correlation high for random points...




