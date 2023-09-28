import matplotlib.pyplot as plt
import pandas as pd
import pybaselines.polynomial
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, FastICA
import numpy as np
from scipy import sparse, stats
from scipy.sparse.linalg import spsolve
from BaselineRemoval import BaselineRemoval
import matplotlib.pyplot as plt
import scipy
import numpy
import sklearn
from pybaselines import polynomial

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
    for index, rowy in spectra.iterrows():
        row = rowy.values.reshape(-1, 1)
        row = row.flatten()
        row_polyfit = pybaselines.polynomial.modpoly(row, poly_order=5)[0]
        # plt.plot(row)
        # plt.plot(row_polyfit)
        # plt.show()
        row = row - row_polyfit
        row = row.flatten()
        row = row.reshape(1, -1)
        normalized_df = pd.DataFrame(row, columns=rowy.index)
        baseline_removed.append(normalized_df)

    baselined_spectra = pd.concat(baseline_removed, axis=0, ignore_index=True)

    return baselined_spectra

def fast_ICA(data, n):
    model = FastICA(n_components= n, whiten='unit-variance')
    columns_ = [i for i in range(n)]
    data = model.fit_transform(data)
    data = pd.DataFrame(data=data, columns=columns_)
    return data

if __name__ == '__main__':
    data1 = pd.read_csv('data/pca_data/allsol_580_BR_NM_3com.csv')
    data2 = pd.read_csv('data/pca_data/PEG_580_BR_NM_3com.csv')
    data = pd.read_csv('data/data_610_BR_NM.csv')
    data3 = pd.read_csv('data/data_580_BR_NM.csv')

    rawdata = pd.read_csv('data/data_580_BR_NM.csv')
    concentrations = pd.read_csv('data/data_580_concentrations.csv')

    x = fast_ICA(data1, 3)
    print(x)
    #plt.plot(data1.iloc[1])
    plt.plot(x.iloc[0])
    plt.show()






    # next steps
    # whiten data for fast ICA



