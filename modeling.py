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
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.cluster import KMeans

"""
Machine Learning!!!!

where the magic happens. 
"""

def find_data_onbadspec(bad_spec, x, y, names):
    """
    can be called in the last line of the evaluate_with_models() function
    finds indices of spectra with really high MAE's
    :param listy: list of spectra that returned error specified by if statement
    :param x: original features
    :param y: original labels
    :param names: names of the spectra in x
    :return: print the indices, concentrations, and names of spectra above a certain error
    """
    x = np.array(x)
    y = np.array(y)

    indices = []
    conc = []
    namey = []
    for i in range(len(bad_spec)):
        row_index = np.where(np.all(x == bad_spec[i], axis=1))[0]
        indices.append(row_index)
        conc.append((y[row_index]))
        namey.append(names.iloc[row_index])
    print('indices:', indices)
    print('concentrations:', conc)
    print('names', namey)
    return None

def evaluate_withmodels(x, y, names, n):
    """
    tests different ML models with the given x and y data. prints the MAE for each model.
    explore this function, because there's a lot that can be changed!
    can save models as .pkl files, change number of iterations, change train/test split,
    get data on bad spec, etc.
    :param x: features as a dataframe where each row is a spectra
    :param y: labels as a dataframe column
    :param names: used for semi-random train test split function. names of raman spec
    :param n: test size (ex: 0.15)
    :return: predicted concentrationvalues
    """

    # models:
    RF = RandomForestRegressor()
    SVM = svm.SVR()
    GBRT = GradientBoostingRegressor(alpha=.001, n_estimators=50000)
    MLP = MLPRegressor(random_state=1, hidden_layer_sizes=(100, ), solver='lbfgs', max_iter=5000)
    KR = KernelRidge()
    KNN = KNeighborsRegressor()
    LR = LinearRegression()

    models = [RF, SVM, MLP, KR, KNN, LR] # models that are evaluated

    badspec = []
    results = []
    for j in models:
        model = j
        i = 0
        listy = []
        while i < 10: # number of iterations- can be changed
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=n) #split_by_concentration(x, y, names, n) # change this to split based on names
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            y_pred = np.maximum(y_pred, 0)

            for m in range(len(y_pred)):
                try:
                    if mean_absolute_error(y_test.iloc[m], y_pred[m]) > 1: # for analyzing which spec are giving high errors
                        badspec.append(list(x_test.iloc[m]))
                except TypeError:
                    if mean_absolute_error(y_test.iloc[m], [y_pred[m]]) > 1: # change to whatever threshold!
                        badspec.append(list(x_test.iloc[m]))

            mae = mean_absolute_error(y_test, y_pred)
            listy.append(mae)
            i += 1
        # joblib.dump(model,'models/GBRT_cut_data_wconc_allsol_580_BR_NM_10com.pkl') # uncover to save model
        results.append([j,'MAE:', np.average(listy)])
        print(results)
    return y_pred #find_data_onbadspec(badspec, x, y, names) # can change this arg- right now it finds which spectra have the highest errors


def split_by_concentration(x, y, names, split):
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
#

def phosphate_training_set(x, y, names, split):
    # Makes a training and test split that has an even amount of PEG, BSA, and phos solvent samples
    # Sorts them by names

    def categorize_row(row):
        if 'BSA' in row:
            return 1
        elif 'PEG' in row:
            return 2
        else:
            return 3

    # Apply categorization to the 'names' column
    names['categories'] = names.applymap(categorize_row)['names']

    # Calculate the number of samples for each category based on the split ratio
    category_counts = names['categories'].value_counts()
    ones_count = int(category_counts[1] * split)
    twos_count = int(category_counts[2] * split)
    threes_count = int(category_counts[3] * split)

    # Filter samples for each category
    ones_samples = names[names['categories'] == 1].head(ones_count)
    twos_samples = names[names['categories'] == 2].head(twos_count)
    threes_samples = names[names['categories'] == 3].head(threes_count)

    # Combine samples for training and testing sets
    training_names = pd.concat([ones_samples, twos_samples, threes_samples])
    testing_names = names.drop(training_names.index)

    # Use the indices from names DataFrame to filter the corresponding rows in x and y
    training_indices = training_names.index
    testing_indices = testing_names.index

    training_x = x.loc[training_indices]
    training_y = y.loc[training_indices]

    testing_x = x.loc[testing_indices]
    testing_y = y.loc[testing_indices]

    return training_x, testing_x, training_y, testing_y

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

def create_index_dataframe(dataframe, threshold):

    # Calculate the correlation matrix
    correlation_matrix = dataframe.corr()

    # Get a boolean mask for the lower triangular part
    mask = np.tri(correlation_matrix.shape[0], dtype=bool)

    # Set values in the lower triangle to NaN
    # correlation_matrix = correlation_matrix.mask(mask)

    # Create an empty DataFrame to store indices
    index_dataframe = pd.DataFrame()

    store = []

    # Iterate over each column
    for column_name in correlation_matrix.columns:
        # Find indices where values are greater than the threshold
        filtered_indices = correlation_matrix.index[correlation_matrix[column_name] > threshold].tolist()
        store.append(filtered_indices)

    max_length = max(len(lst) for lst in store)
    print("Maximum length:", max_length)

    for i in range(len(store)):
        # Ensure all columns have the same size, fill with NaN if necessary
        if len(store[i]) < max_length:
            store[i].extend([None] * (max_length - len(store[i])))

        # Add a new column to the index_dataframe with column_name as the header
        index_dataframe[i] = pd.Series(store[i])

    return index_dataframe

def calculate_iou(dataframe, column1, column2):
    # Extract unique values from the specified columns
    values_column1 = set(dataframe[column1])
    values_column2 = set(dataframe[column2])

    # Calculate Intersection and Union
    intersection = values_column1.intersection(values_column2)
    union = values_column1.union(values_column2)

    # Calculate IoU
    iou = len(intersection) / len(union)

    return iou

def reduce_components(df):

    for column in df.columns:
        for column2 in df.columns:
            if column != column2 and calculate_iou(df, column, column2) > 0.8:
                # Combine columns with high IoU
                combined_column = pd.Series(pd.concat([df[column], df[column2]]).unique())

                # Add new combined column to the DataFrame
                new_column_name = f"{column}_{column2}"
                df[new_column_name] = combined_column

                # Drop the marked columns
                df = df.drop(columns=[column, column2], errors='ignore')
                print(df)
                return reduce_components(df)

    return df


if __name__ == '__main__':
    x1 = pd.read_csv('data/integrated spectra/150gg_data_GSH_20.csv')
    y1 = pd.read_csv('data/GSH_conc_150gg_data.csv')
    names = pd.read_csv('data/raman_580_names.csv')
    df = pd.read_csv('data/correlation analysis/prepro_corr_points_580.csv')

    evaluate_withmodels(x1, y1, names, .2)
    exit()


