import matplotlib.pyplot as plt
import pandas as pd
import pybaselines.polynomial
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, FastICA
from scipy.signal import savgol_filter

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
    vectors = pca.components_[1]
    pca_Df = pd.DataFrame(data=pca_components, columns=columns_)
    return pca_Df, ratio, vectors


def remove_baseline(spectra):
    # polynomial fitting for baseline removal

    baseline_removed = []
    for index, rowy in spectra.iterrows():
        row = rowy.values.reshape(-1, 1)
        row = row.flatten()
        row_polyfit = pybaselines.polynomial.modpoly(row, poly_order=8)[0]
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

def smooth_spectra(data):
    # Parameters:
    w = 9  # window (number of points)
    p = 2  # polynomial order

    # polynomial fitting for baseline removal

    baseline_removed = []
    for index, rowy in data.iterrows():
        row = rowy.values.reshape(-1, 1)
        row = row.flatten()
        row_polyfit = savgol_filter(row, w, polyorder = p, deriv=0)
        # plt.plot(row)
        # plt.plot(row_polyfit)
        # plt.show()
        row = row_polyfit.flatten()
        row = row.reshape(1, -1)
        normalized_df = pd.DataFrame(row, columns=rowy.index)
        baseline_removed.append(normalized_df)

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

if __name__ == '__main__':
    data1 = pd.read_csv('data/pca_data/allsol_580_BR_NM_3com.csv')
    data2 = pd.read_csv('data/pca_data/PEG_580_BR_NM_3com.csv')
    data = pd.read_csv('data/data_610_BR_NM.csv')
    data3 = pd.read_csv('data/data_580_BR_NM.csv')

    rawdata = pd.read_csv('data/PEGdata_580_BR_NM.csv')
    subspec = pd.read_csv('data/cut_data_580_BR_NM.csv')
    names = pd.read_csv('data/data_580_names.csv')
    concentrations = pd.read_csv('data/cut_BSAdata_580_BR_NM_concentrations_GSSG.csv')

    for i in range(len(subspec)):
        plt.plot(subspec.iloc[i])
        plt.title(str(names.iloc[i]))
        plt.show()

    subtract_solventspec(rawdata, subspec)




