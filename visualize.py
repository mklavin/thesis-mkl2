import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from data_collection import is_nan_string
import seaborn as sns
from sklearn.cluster import DBSCAN, KMeans
from umap import UMAP
import pybaselines.polynomial

def separate_by_sol_andplot(data, indices):
    transposed_rows = []  # Collect transposed rows in a list

    for i in indices:
        if is_nan_string(i):
            continue  # Skip NaN strings

        i = int(i)
        rowy = data.iloc[i]
        rowy = rowy.T
        transposed_rows.append(rowy)

    # Concatenate all transposed rows into the output DataFrame
    if transposed_rows:
        output = pd.concat(transposed_rows, axis=1)
    else:
        output = pd.DataFrame()  # Empty DataFrame if no valid rows were found

    return output.T

def plot_and_cluster_DBSCAN(dataframe):
    clustering = DBSCAN(eps=1.5).fit(dataframe)

    # uncover to make scatterplot
    scatter_plot = sns.scatterplot(data=dataframe, x=dataframe['0'], y=dataframe['1'], alpha=0.3, palette='gist_ncar',
                    hue_order=np.random.shuffle(np.arange(len(clustering.labels_))),
                    hue=clustering.labels_).set_title(f"Neighbors= {40}, eps=5")
    sns.set(font_scale=2)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.xlabel('UMAP1', fontsize=16)
    plt.ylabel('UMAP2', fontsize=16)
    plt.title(label=f"Clustering on 10-D UMAP Values", fontsize=20)
    scatter_fig = scatter_plot.get_figure()
    scatter_fig.savefig('graph2.png', dpi= 1200)
    plt.show()
    return clustering.labels_

def plot_and_cluster_kmeans(dataframe):
    kmeans = KMeans(n_clusters=3)
    clustering = kmeans.fit(dataframe)

    # uncover to make scatterplot
    scatter_plot = sns.scatterplot(data=dataframe, x=dataframe['0'], y=dataframe['1'], alpha=0.3,
                                   hue_order=np.random.shuffle(np.arange(len(clustering.labels_))),
                                   hue=clustering.labels_).set_title(f"Neighbors= {40}, eps=5")
    sns.set(font_scale=2)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.xlabel('UMAP1', fontsize=16)
    plt.ylabel('UMAP2', fontsize=16)
    plt.title(label=f"Clustering on 10-D UMAP Values", fontsize=20)
    scatter_fig = scatter_plot.get_figure()
    scatter_fig.savefig('graph2.png', dpi=1200)
    plt.show()
    return clustering.labels_

def make_thesisplot(x, y):
    row_polyfit = pybaselines.polynomial.imodpoly(y, poly_order=10)[0]

    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize='x-small')
    plt.rc('ytick', labelsize='x-small')

    fig = plt.figure()
    ax = fig.add_subplot(2, 2, 1)

    ax.plot(x, y, color='b', ls='solid')
    ax.plot(x, row_polyfit, color='r', ls='solid')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Temperature (K)')

    ax2 = fig.add_subplot(2, 2,2)
    ax2.plot(x, y-row_polyfit)
    plt.show()
    return None


if __name__ == '__main__':
    data = pd.read_csv('data/pca_data/allsol_580_BR_NM_10com.csv')
    data2 = pd.read_csv('data/data_610_BR_NM.csv')
    soldata = pd.read_csv('data/data_580.csv')

    make_thesisplot(np.arange(0, len(soldata.iloc[5])), soldata.iloc[5])
    exit()

    # uncover to plot different solvents
    bsadf = separate_by_sol_andplot(data, soldata['BSA'])
    pegdf = separate_by_sol_andplot(data, soldata['PEG'])
    phosdf = separate_by_sol_andplot(data, soldata['phos'])
    plt.scatter(bsadf['0'], bsadf['1'], c='g')
    plt.scatter(pegdf['0'], pegdf['1'])
    plt.scatter(phosdf['0'], phosdf['1'])
    plt.show()

    plot_and_cluster_kmeans(data)

    # try:
    # taking a smaller snapshot of spectrum around GSH/GSSG peaks- try with PCA and maybe even models
    # 121 -213? - 580
    # 670 -834? - 610
    # 1300 dimension PCA
    # subtract and do two dimensions
    # FIND VECTORS ASSOCIATED WITH THE PRINCIPAL COMPONENTS


