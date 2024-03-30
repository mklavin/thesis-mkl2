import matplotlib.pyplot as plt
import pandas as pd
import pybaselines.spline
from sklearn.decomposition import PCA, FastICA
from sklearn.preprocessing import StandardScaler
from scipy.signal import savgol_filter, argrelmin
import numpy as np

"""
ALL PREPROCESSING METHODS

- contains functions for standardizing spectra, normalizing, 
  baseline removal, PCA, spectral smoothing
- these functions are generally not utilized directly- mainly used in the strategy_search.py file
"""

def standardize_byscalar(spectra, scalar):
    """
    :param spectra: row of dataframe
    :param scalar: scalar to multiply all the spectra by
    :return: row of databrame multiplied by a scalar
    """

    spectra_scaled = spectra * scalar
    # uncover to visualize scaling
    # plt.plot(np.linspace(0,len(spectra),len(spectra)),spectra)
    # plt.plot(np.linspace(0, len(spectra), len(spectra)), spectra_scaled)
    # plt.show()
    return spectra_scaled

def integrate_dataframe(df, increment_size=10):
    integrated_df = pd.DataFrame()

    # Iterate over each row in the DataFrame
    for index, row in df.iterrows():
        y_values = row.values  # Extract y values from the row

        # Calculate the number of increments
        num_increments = len(y_values) // increment_size

        # Initialize a new vector for the integrated values
        integrated_values = np.zeros(num_increments)

        # Integrate each increment of specified size
        for i in range(num_increments):
            start_index = i * increment_size
            end_index = (i + 1) * increment_size
            integrated_values[i] = np.trapz(y_values[start_index:end_index])

        # Create a new DataFrame with the integrated values for the current row
        integrated_df = pd.concat([integrated_df, pd.DataFrame(integrated_values).transpose()], ignore_index=True)

    return integrated_df

def PCA1(data, n):
    """
    :param data: dataframe of raman spectra, mean zero std one
    :param n: number of components for PCA analysis
    :return: PCA dataframe
    """
    columns_ = [i for i in range(n)]
    pca = PCA(n_components=n)
    pca = pca.fit(data)
    ratio = pca.explained_variance_ratio_
    pca_components = pca.transform(data)
    pca_Df = pd.DataFrame(data=pca_components, columns=columns_)
    return pca_Df, sum(ratio)

def fast_ICA(data, n):
    model = FastICA(n_components= n, whiten='unit-variance')
    columns_ = [i for i in range(n)]
    data = model.fit_transform(data)
    data = pd.DataFrame(data=data, columns=columns_)
    return data

def subtract_solventspec(data, solventspec):
    """
    :param data: dataframe of raman spectra
    :param solventspec: spectra to subtract from the dataframe
    :return: dataframe after subtraction
    """
    baseline_removed = []
    for index, rowy in data.iterrows():
        row = rowy.values.reshape(-1, 1)
        row = row.flatten()
        # plt.plot(row)
        # plt.plot(solventspec)
        # row = row - solventspec
        # plt.show()
        # print(row)
        #row = row.reshape(1, -1)
        normalized_df = pd.DataFrame(row, columns=rowy.index)
        baseline_removed.append(normalized_df)

    baselined_spectra = pd.concat(baseline_removed, axis=0, ignore_index=True)

    return baselined_spectra

def scale_rows_to_max(dataframe, region:str):
    """
    :param dataframe: dataframe of raman spectra
    :param region: 580 or 610 spectral region, determines where to search for the water peak
    :return: dataframe of raman spectra scaled to the water peak
    """

    # Calculate the maximum value in each row
    if region == '610':
        max_values = dataframe.iloc[:, 140:170].max(axis=1)
    if region == '580':
        max_values = dataframe.iloc[:, 140:170].max(axis=1)

    scaled_dataframe = dataframe.mul(20/max_values, axis=0)

    return scaled_dataframe

def smooth_spectra(data):
    """
    :param data: dataframe of raman spectra
    :return: smoothed dataframe
    """
    # Parameters:
    w = 9  # window (number of points)
    p = 2  # polynomial order

    # polynomial fitting for baseline removal

    i=0
    baseline_removed = []
    for index, rowy in data.iterrows():
        row = rowy.values.reshape(-1, 1)
        row = row.flatten()
        row_polyfit = savgol_filter(row, w, polyorder = p, deriv=0)
        # plt.plot(row)
        # plt.plot(row_polyfit)
        # plt.title(str(i))
        # plt.show()
        row = row_polyfit.flatten()
        row = row.reshape(1, -1)
        normalized_df = pd.DataFrame(row, columns=rowy.index)
        baseline_removed.append(normalized_df)
        i += 1

    baselined_spectra = pd.concat(baseline_removed, axis=0, ignore_index=True)

    return baselined_spectra

def normalize(spectra):
    """
    :param spectra: dataframe of raman spectra
    :return: normalized rows of dataframe
    """
    normalized = []

    for index, rowy in spectra.iterrows():
        row = rowy.values.reshape(-1, 1)
        row = StandardScaler().fit_transform(row)
        row = row.flatten()
        row = row.reshape(1, -1)
        normalized_df = pd.DataFrame(row, columns=rowy.index)
        normalized.append(normalized_df)
    normalized_spectra = pd.concat(normalized, axis=0, ignore_index=True)

    return normalized_spectra

def remove_baseline(spectra, baseline_func, order=None):
    """
    :param spectra: dataframe of raman spectra
    :param baseline_func: pybaselines function to use
    :return: dataframe of raman spectra with baselines subtracted
    """
    # polynomial fitting for baseline removal

    baseline_removed = []
    i = 0
    for index, rowy in spectra.iterrows():
        row = rowy.values.reshape(-1, 1)
        row = row.flatten()
        if order is not None:
            row_polyfit = baseline_func(row, poly_order=order)[0]
        else:
            row_polyfit = baseline_func(row)[0]
        # plt.title(str(i))
        # plt.plot(row)
        # plt.plot(row_polyfit)
        # plt.show()
        row = row - row_polyfit
        row = row.flatten()
        row = row.reshape(1, -1)
        normalized_df = pd.DataFrame(row, columns=rowy.index)
        baseline_removed.append(normalized_df)
        i += 1

    baselined_spectra = pd.concat(baseline_removed, axis=0, ignore_index=True)

    return baselined_spectra

def calc_correlation_matrix(df, conc):
    # return the correlation matrix of all points

    df = pd.concat([df, conc], axis=1) # if you want to append concentrations
    correlation_matrix = pd.DataFrame(np.corrcoef(df.T))

    return correlation_matrix



if __name__ == '__main__':
    df = pd.read_csv('data/150gg_data_prepro.csv')
    glu = pd.read_csv('data/correlation analysis/prepro_corr_glu_580.csv')
    conc = pd.read_csv('data/GSSG_conc_150gg_data.csv')
    names = pd.read_csv('data/raw data/daniels_data/danielmimi_data_580_names.csv')

    df = calc_correlation_matrix(df, conc)
    df.to_csv('data/correlation analysis/150gg_data_prepro_GSSG.csv', index=False)

    exit()


    xpoints = []
    ypoints = []

    for i in range(len(glu)):
        if glu.iloc[i]['153'] > 0.7:
            xpoints.append(glu.iloc[i]['point'])
            point = glu.iloc[i]['point']
            ypoints.append(df.iloc[3][int(point)])

    # Sort the x-points to ensure they are in ascending order
    sorted_indices = sorted(range(len(xpoints)), key=lambda k: xpoints[k])
    xpoints = [xpoints[i] for i in sorted_indices]
    ypoints = [ypoints[i] for i in sorted_indices]

    plt.plot(df.iloc[3], label='Line Plot')
    plt.plot(xpoints, ypoints, 'ro', markersize=3, label='Points')
    plt.legend()
    plt.show()















