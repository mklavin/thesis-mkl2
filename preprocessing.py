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
        plt.plot(row)
        plt.plot(solventspec)
        row = row - solventspec
        plt.show()
        print(row)
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
    # smooth, remove baseline, df = df.iloc[:, 120:250], normalize, PCA var > 99.5%

    df = smooth_spectra(df)
    df = remove_baseline(df)
    df = df.iloc[:, 744:831]
    df = normalize(df)

    for i in range(len(df.iloc[0])):
        plt.plot(df.iloc[i])
        plt.show()

    for i in range(40):
        pca_Df, ratio = PCA1(df, i)
        if sum(ratio) > 0.995:
            print(sum(ratio),i)
            break
    return pca_Df

def preprocess2(df):
    # smooth, df = df.iloc[:, 120:250], remove baseline, normalize, PCA var > 99.5%

    df = smooth_spectra(df)
    df = df.iloc[:, 120:250]
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

def preprocess3(df):
    # smooth, remove first peak, remove baseline, df = df.iloc[:, 120:200], normalize, PCA var > 99.5%

    df = df.iloc[:, 50:]
    df = smooth_spectra(df)
    df = remove_baseline(df)
    df = df.iloc[:, 120:250]
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

def preprocess4(df):
    # smooth, remove baseline, df = df.iloc[:, 120:250], normalize, PCA var > 99.5%

    df = smooth_spectra(df)
    df = df.iloc[:, 120:250]
    df = remove_baseline(df)
    #df = df.iloc[:, 62:171]
    df = normalize(df)

    # for i in range(len(df.iloc[0])):
    #     plt.plot(df.iloc[i])
    #     plt.show()

    for i in range(40):
        data, ratio = PCA1(df, i)
        if sum(ratio) > 0.995:
            print(sum(ratio),i)
            break
    return data

def integrate_rows(dataframe):
    integration_list = []

    for index, row in dataframe.iterrows():
        x = np.arange(len(row))
        y = row
        area_under_curve = np.trapz(y, x)
        integration_list.append(area_under_curve)

    integration_df = pd.DataFrame({'Integration': integration_list})

    return integration_df



if __name__ == '__main__':
    df = pd.read_csv('data/data_580.csv')
    conc = pd.read_csv('data/data_610_concentrations_GSH.csv')

    # plt.plot(df.iloc[130])
    # plt.show()

    # how well does model perform when bad spectra are dropped?
    # bad_spec = [4,5,9,8,10,13,15,16,17,19,20,22,34,39,40,51,53,55,59,64,68,73,74,80,87,97]
    # cutdf = df.drop(bad_spec)
    # cutconc = conc.drop(bad_spec)

    df = scale_rows_to_max(df)

    df = preprocess4(df)


    df.to_csv('data/prepro_methods/allsol_580_pre4_27com.csv', index=False)





