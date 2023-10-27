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
    # Fit a polynomial to the data
    polyfit_order = 7
    row_polyfit = pybaselines.polynomial.imodpoly(y, poly_order=polyfit_order)[0]

    # Customize plot styles for better readability
    plt.rc('font', family='serif', size=12)
    plt.rc('xtick', labelsize='small')
    plt.rc('ytick', labelsize='small')

    # Create a figure and add subplots in a 2x1 grid
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))  # Adjust the figure size to your preference

    # Plot the data and the polynomial fit in the upper subplot
    ax1.plot(x, y, color='b', linestyle='solid', label='Data')
    ax1.plot(x, row_polyfit, color='r', linestyle='solid', label=f'Polynomial Fit (Order {polyfit_order})')
    ax1.set_xlabel('Raman Shift (cm⁻¹)')
    ax1.set_ylabel('Intensity')
    ax1.legend()  # Add a legend to the upper subplot

    # Plot the residual (data - fit) in the lower subplot
    residual = y - row_polyfit
    ax2.plot(x, residual, color='g', linestyle='solid', label='Residual')
    ax2.set_xlabel('Raman Shift (cm⁻¹)')
    ax2.set_ylabel('Intensity')
    ax2.legend()  # Add a legend to the lower subplot

    # Save the plot to a file for inclusion in your paper
    plt.savefig('thesis_plot.png', dpi=300, bbox_inches='tight')  # Adjust the file format and resolution as needed

    # Display the plot (optional)
    plt.show()

    return None  # This line is not needed

if __name__ == '__main__':
    data = pd.read_csv('data/pca_data/allsol_580_BR_NM_10com.csv')
    data2 = pd.read_csv('data/data_610_BR_NM.csv')
    soldata = pd.read_csv('data/data_580.csv')

    make_thesisplot(np.arange(250, len(soldata.iloc[5])+250), soldata.iloc[5])
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


