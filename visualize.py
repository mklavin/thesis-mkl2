import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from data_collection import is_nan_string
import seaborn as sns
from sklearn.cluster import DBSCAN 

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

def plot_and_cluster(dataframe, nneighbors):
    """
    :param dataframe: features
    :param nneighbors: UMAP parameter input
    :return: plots UMAP values and clusters based on features, also returns clustering labels
    """
    if 'SMILES' in dataframe.columns:
        dataframe = dataframe.drop(columns='SMILES')
    if 'Name' in dataframe.columns:
        dataframe = dataframe.drop(columns='Name')

    dfs = pd.DataFrame(UMAP(n_components=10, n_neighbors=nneighbors, min_dist=0.1).fit_transform(dataframe),
                                index=dataframe.index,
                                columns=["UMAP1", "UMAP2","UMAP3","UMAP4","UMAP5","UMAP6",
                                         "UMAP7","UMAP8","UMAP9","UMAP10"])

    clustering = DBSCAN(eps=.5).fit(dfs)

    # add noise to ligand_files
    #umap1 = np.array(dataframe['UMAP1'])
    #umap2 = np.array(dataframe['UMAP2'])
    #for i in range(len(umap1)):
        #randy = random.uniform(-1,1)
        #umap1[i] = umap1[i] + randy
    #for i in range(len(umap2)):
        #randy = random.uniform(-1,1)
        #umap2[i] = umap2[i] + randy

    # uncover to make scatterplot
    scatter_plot = sns.scatterplot(data=dataframe, x=dataframe['UMAP1'], y=dataframe['UMAP2'], alpha=0.3, palette='gist_ncar',
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

if __name__ == '__main__':
    pcadata = pd.read_csv('data/pca_data/allsol_610_BR_NM_3com.csv')
    soldata = pd.read_csv('data/separate_by_sol_610.csv')

    bsadf = separate_by_sol_andplot(pcadata, soldata['BSA'])
    pegdf = separate_by_sol_andplot(pcadata, soldata['PEG'])
    phosdf = separate_by_sol_andplot(pcadata, soldata['phos'])

    plt.scatter(bsadf['0'], bsadf['1'])
    plt.scatter(pegdf['0'], pegdf['1'])
    plt.scatter(phosdf['0'], phosdf['1'])
    plt.show()
