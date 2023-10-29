import matplotlib.pyplot as plt
import pandas as pd
import pybaselines.polynomial
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, FastICA
from scipy.signal import savgol_filter, argrelmin
import numpy as np
from scipy.signal import find_peaks

def standardize_byscalar(spectra, scalar):
    # multiply all values by a scalar
    # used to standardize spectra

    spectra_scaled = spectra * scalar
    # uncover to visualize scaling
    # plt.plot(np.linspace(0,len(spectra),len(spectra)),spectra)
    # plt.plot(np.linspace(0, len(spectra), len(spectra)), spectra_scaled)
    # plt.show()
    return spectra_scaled

def normalize(spectra):
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


def PCA1(data, n):
    columns_ = [i for i in range(n)]
    pca = PCA(n_components=n)
    pca = pca.fit(data)
    ratio = pca.explained_variance_ratio_
    pca_components = pca.transform(data)
    pca_Df = pd.DataFrame(data=pca_components, columns=columns_)
    return pca_Df, ratio


def remove_baseline(spectra):
    # polynomial fitting for baseline removal

    baseline_removed = []
    i = 0
    for index, rowy in spectra.iterrows():
        row = rowy.values.reshape(-1, 1)
        row = row.flatten()
        row_polyfit = pybaselines.polynomial.imodpoly(row, poly_order=8)[0]
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

def fast_ICA(data, n):
    model = FastICA(n_components= n, whiten='unit-variance')
    columns_ = [i for i in range(n)]
    data = model.fit_transform(data)
    data = pd.DataFrame(data=data, columns=columns_)
    return data

def smooth_spectra(data):
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

def subtract_solventspec(data, solventspec):
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

def scale_rows_to_max(dataframe):
    scaled_dataframe = dataframe.copy()  # Make a copy of the input DataFrame to avoid modifying it

    # Calculate the maximum value in each row
    max_values = dataframe.max(axis=1)

    # Scale each row to have the same maximum value (target_max)
    scaled_dataframe = scaled_dataframe.mul(20 / max_values, axis=0)

    return scaled_dataframe

def preprocess1(df):
    # smooth, remove baseline, normalize, PCA var > 99.5%

    df = smooth_spectra(df)
    df = remove_baseline(df)
    df = normalize(df)

    # for i in range(len(df.iloc[0])):
    #     plt.plot(df.iloc[i])
    #     plt.show()

    for i in range(40):
        pca_Df, ratio = PCA1(df, i)
        if sum(ratio) > 0.995:
            print(sum(ratio),i)
            break
    return pca_Df


if __name__ == '__main__':
    df = pd.read_csv('data/data_610.csv')
    conc = pd.read_csv('data/data_580_concentrations_GSSG.csv')
    names = pd.read_csv('data/danielmimi_data_580_names.csv')

    plt.plot(np.arange(0, len(df.iloc[3])), df.iloc[3])
    plt.show()
    exit()

    df = smooth_spectra(df)
    df = remove_baseline(df)
    df = normalize(df)

    total = pd.concat([df, conc], axis=1)

    df = pd.DataFrame(np.corrcoef(total.T))
    # Add an index column
    df.reset_index(inplace=True)

    # Sort column 'A' in ascending order while preserving the index
    sorted_df = df.sort_values(by=1338, ascending=False)
    sorted_df = sorted_df[['index', 1338]]

    sorted_df.to_csv('data/corr_anal_580.csv', index=False)
    exit()
    # how well does model perform when bad spectra are dropped?
    # bad_spec = [4,5,9,8,10,13,15,16,17,19,20,22,34,39,40,51,53,55,59,64,68,73,74,80,87,97]
    # cutdf = df.drop(bad_spec)
    # cutconc = conc.drop(bad_spec)

    plt.plot(df.iloc[2])
    plt.show()
    exit()


    df = preprocess5(df)
    df.to_csv('data/prepro_methods/danielmimi_580_pre4_27com.csv', index=False)





