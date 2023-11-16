import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from data_collection import is_nan_string
import seaborn as sns
from sklearn.cluster import DBSCAN, KMeans
from umap import UMAP
import pybaselines.polynomial
from collections import Counter

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

def make_baselineplot(x, y):
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

def make_sampledist_plot(conc):
    # Extract the data and compute value counts
    x = conc['conc_GSSG']
    x_counter = x.value_counts()

    # Set up Seaborn for improved aesthetics
    sns.set(style="whitegrid", font_scale=1.2)

    # Create the horizontal bar plot
    plt.figure(figsize=(8, 6))  # Set the figure size
    ax = x_counter.plot(kind='barh', color='hotpink')

    # Customize the plot labels and title
    ax.set_xlabel('Number of Samples', labelpad=15)  # X-axis label
    ax.set_ylabel('GSSG Concentration (mM)', labelpad=15)  # Y-axis label
    ax.set_title('Distribution of Sample Concentrations in the 580 Region', pad=20)  # Title

    # Customize ticks and labels
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))  # Show only integer ticks on the x-axis
    plt.xticks(fontsize=12)  # X-axis tick font size
    plt.yticks(fontsize=12)  # Y-axis tick font size

    # Remove top and right spines
    sns.despine()

    # Save the plot as a high-resolution image (optional)
    plt.savefig("sample_distribution_plot580.png", dpi=300, bbox_inches='tight')

    # Display the plot
    plt.show()

    return None

def make_barplot_concetrations(df):
    # Count the number of values in each column
    value_counts = df.count()

    # Create a bar plot
    plt.figure(figsize=(8, 6))
    value_counts.plot(kind='bar', color='hotpink')

    # Customize plot labels and title
    plt.xlabel('Solvents', labelpad=15)
    plt.ylabel('Number of Samples', labelpad=15)
    plt.title('Number of Samples in the 610 Region for Each Solvent Type', pad=20)

    # Customize tick labels
    plt.xticks(range(len(df.columns)), df.columns, rotation=0, fontsize=12)
    plt.yticks(fontsize=12)

    # Save the plot as a high-resolution image (optional)
    plt.savefig("solvent_distribution_plot610.png", dpi=300, bbox_inches='tight')

    # Display the plot
    plt.show()

    return None

def plot_predicted_versus_test(y_pred, y_test):
    # Scatter plot
    plt.scatter(y_pred, y_test, color='blue', marker='o', label='Actual vs. Predicted')

    # Diagonal line for reference
    plt.plot(np.arange(0, 90), np.arange(0, 90), color='red', linestyle='--', label='Ideal Line')

    # Adding labels and title
    plt.xlabel('Predicted Values')
    plt.ylabel('Actual Values')
    plt.title('Actual vs. Predicted Values of GSH (mM)')

    # Displaying the legend
    plt.legend()

    # Adding grid for better readability
    plt.grid(True)
    plt.savefig('predvsactual_GSH.png')

    # Show the plot
    plt.show()

    return None

if __name__ == '__main__':
    data = pd.read_csv('data/pca_data/allsol_580_BR_NM_10com.csv')
    data2 = pd.read_csv('data/prepro_methods/data_610_BR_NM.csv')
    soldata = pd.read_csv('data/data_580.csv')
    conc = pd.read_csv('data/data_580_concentrations_GSSG.csv')
    sol = pd.read_csv('data/separate_by_sol_610.csv')


    make_barplot_concetrations(sol)
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


