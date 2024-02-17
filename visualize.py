import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from data_collection import is_nan_string
import seaborn as sns
from sklearn.cluster import DBSCAN, KMeans
from umap import UMAP
import pybaselines.polynomial
from collections import Counter
from preprocessing import PCA1, scale_rows_to_max

# MOST PLOTS ARE PARTIALLY AI GENERATED VIA CHATGPT

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
    dataframe, ratio = PCA1(dataframe, 2)
    print(ratio)
    clustering = DBSCAN(eps=60).fit(dataframe)

    # uncover to make scatterplot
    scatter_plot = sns.scatterplot(data=dataframe, x=dataframe[0], y=dataframe[1], alpha=0.1, palette='gist_ncar',
                    hue_order=np.random.shuffle(np.arange(len(clustering.labels_))),
                    hue=clustering.labels_).set_title(f"Neighbors= {5}, eps=5")
    sns.set(font_scale=2)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.xlabel('UMAP1', fontsize=16)
    plt.ylabel('UMAP2', fontsize=16)
    #plt.title(label=f"Clustering on 10-D UMAP Values", fontsize=20)
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
    row_polyfit = pybaselines.morphological.jbcd(y)[0]

    # Customize plot styles for better readability
    plt.rc('font', family='serif', size=12)
    plt.rc('xtick', labelsize='small')
    plt.rc('ytick', labelsize='small')

    # Create a figure and add subplots in a 2x1 grid
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))  # Adjust the figure size to your preference

    # Plot the data and the polynomial fit in the upper subplot
    ax1.plot(x, y, color='b', linestyle='solid', label='Data')
    ax1.plot(x, row_polyfit, color='r', linestyle='solid', label='Joint Baseline Correction and Denoising')
    ax1.set_xlabel('Raman Shift (cm⁻¹)')
    ax1.set_ylabel('Intensity')
    ax1.legend()  # Add a legend to the upper subplot

    # Plot the residual (data - fit) in the lower subplot
    residual = y - row_polyfit
    ax2.plot(x, residual, color='g', linestyle='solid', label='Residual')
    ax2.set_xlabel('Raman Shift (cm⁻¹)')
    ax2.set_ylabel('Intensity')
    ax2.legend()  # Add a legend to the lower subplot

    # Set x-axis ticks to display integers for both subplots
    x_ticks_positions = np.arange(0, 1340, 223.3)
    x_ticks_labels = np.arange(400, 1740, 200)
    ax1.set_xticks(x_ticks_positions)
    ax1.set_xticklabels(x_ticks_labels, fontsize=12)
    ax2.set_xticks(x_ticks_positions)
    ax2.set_xticklabels(x_ticks_labels, fontsize=12)

    # Save the plot to a file for inclusion in your paper
    plt.savefig('baseline_removal_plot.png', dpi=300, bbox_inches='tight')  # Adjust the file format and resolution as needed

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
    plt.title('Actual vs. Predicted Values of GSSG (mM)')

    # Displaying the legend
    plt.legend()

    # Adding grid for better readability
    plt.grid(True)
    plt.savefig('plots/phosphate_model.png')

    # Show the plot
    plt.show()

    return None

def make_solvent_comparison_plot(BSA, PEG, phos):
    # Define colors for each line
    colors = ['blue', 'green', 'red']

    x = np.arange(400, len(BSA)+400)

    # Plot each spectrum with a specific color
    plt.plot(BSA, label='BSA', color=colors[0])
    plt.plot(PEG, label='PEG', color=colors[1])
    plt.plot(phos, label='phos', color=colors[2])

    plt.gcf().set_size_inches(10, 5)

    # Add labels and title
    plt.xlabel('Raman Shift (cm⁻¹)', fontsize = 12)
    plt.ylabel('Intensity', fontsize = 12)
    plt.title('Solvent Comparison of Raman Spectra', fontsize = 12)

    # Customize ticks and labels
    plt.xticks(fontsize=12)  # X-axis tick font size
    plt.yticks(fontsize=12)  # Y-axis tick font size

    # Set x-axis ticks to display integers
    x_ticks_positions = np.arange(400, 2800, 300)
    x_ticks_labels = np.arange(400, 2800, 300)  # [str(int(pos)) for pos in x_ticks_positions]
    plt.xticks(x_ticks_positions, x_ticks_labels, fontsize=12)

    # Add legend
    plt.legend()

    # Save figure
    plt.savefig('solvent_comparison.png', dpi=1200, bbox_inches='tight')

    # Show the plot
    plt.show()

    return None

def correlated_points_on_normal_spec(spec, corr_points):
    # Plot the 1D vector
    plt.plot(spec, linestyle='-', color='b', label='1D Vector')

    # Highlight specific points
    plt.scatter(corr_points, [spec[i] for i in corr_points], color='r', label='Highlighted Points')

    # Add labels and legend
    plt.xlabel('Index')
    plt.ylabel('Values')
    plt.legend()

    # Show the plot
    plt.show()

    return None

def plot_preprocessing_results():
    # Data
    strategies = ['Strategy 1', 'Strategy 2', 'Strategy 3', 'Strategy 4', 'Strategy 5', 'Strategy 6']
    values = [0.6372749936062636, 0.1638503570412616, 0.6201239413631636, 0.6585936251828202, 0.843863308666424, 0.49164083407327897]

    # Create a scatter plot with custom styling
    plt.figure(figsize=(8, 6))  # Set the figure size

    # Customize scatter plot appearance
    plt.scatter(strategies, values, color='skyblue', s=100, edgecolors='black', alpha=0.7, marker='o')

    # Set labels and title with increased font size
    plt.ylabel('Covariance', fontsize=14)
    plt.title('Effect of Strategy Choice on Covariance', fontsize=16)

    # Tilt x-axis labels vertically
    plt.xticks(rotation='vertical')

    # Save figure
    plt.savefig('plots/strategy_comparison.png', dpi=1200, bbox_inches='tight')

    # Show the plot
    plt.show()

    return None
def plot_preprocessing_beforeandafte(before, after):
    # # Customize plot styles for better readability
    # plt.rc('font', family='serif', size=12)
    # plt.rc('xtick', labelsize='small')
    # plt.rc('ytick', labelsize='small')

    # Create a figure and add subplots in a 2x1 grid
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))  # Adjust the figure size to your preference

    # Plot the data and the polynomial fit in the upper subplot
    ax1.plot(np.arange(0, len(before.iloc[0])), before.iloc[0], color='b', linestyle='solid', label='GSSG in Phosphate')
    ax1.plot(np.arange(0, len(before.iloc[0])), before.iloc[4], color='r', linestyle='solid', label='GSSG in BSA')
    ax1.plot(np.arange(0, len(before.iloc[0])), before.iloc[1], color='g', linestyle='solid', label='GSSG in PEG')
    ax1.set_xlabel('Raman Shift (cm⁻¹)')
    ax1.set_ylabel('Intensity')
    ax1.legend()  # Add a legend to the upper subplot

    # Plot the residual (data - fit) in the lower subplot
    ax2.plot(np.arange(0, len(after.iloc[0])), after.iloc[0], color='b', linestyle='solid', label='GSSG in Phosphate')
    ax2.plot(np.arange(0, len(after.iloc[0])), after.iloc[4], color='r', linestyle='solid', label='GSSG in BSA')
    ax2.plot(np.arange(0, len(after.iloc[0])), after.iloc[1], color='g', linestyle='solid', label='GSSG in PEG')
    ax2.set_xlabel('Raman Shift (cm⁻¹)')
    ax2.set_ylabel('Intensity')
    ax2.legend()  # Add a legend to the lower subplot

    # Set x-axis ticks to display integers for both subplots
    x_ticks_positions = np.arange(0, 1340, 223.3)
    x_ticks_labels = np.arange(400, 1740, 200)
    ax1.set_xticks(x_ticks_positions)
    ax1.set_xticklabels(x_ticks_labels, fontsize=12)
    ax2.set_xticks(x_ticks_positions)
    ax2.set_xticklabels(x_ticks_labels, fontsize=12)

    # Save the plot to a file for inclusion in your paper
    plt.savefig('preprocessing_comparison.png', dpi=300, bbox_inches='tight')  # Adjust the file format and resolution as needed

    # Display the plot (optional)
    plt.show()

    return None

def plot_just_solvent(data):
    # Customize plot styles for better readability
    plt.rc('font', family='serif', size=12)
    plt.rc('xtick', labelsize='small')
    plt.rc('ytick', labelsize='small')
    plt.xlabel('Raman Shift (cm⁻¹)')
    plt.ylabel('Intensity')

    x_values = np.arange(0, len(data.iloc[33]))
    y_values = data.drop(columns=[str(i) for i in range(134, 163)], axis=1)
    x_values = np.concatenate((x_values[:133], x_values[162:]))

    # Plot the entire spectrum in blue
    plt.plot(x_values, y_values.iloc[33], color='blue', label='BSA Signal')

    # Show legend
    plt.legend()
    # Save the plot to a file for inclusion in your paper
    plt.savefig('plots/justsolvent.png', dpi=300, bbox_inches='tight')  # Adjust the file format and resolution as needed


    plt.show()

    return None

def plot_just_glutathione(data):
    # Customize plot styles for better readability
    plt.rc('font', family='serif', size=12)
    plt.rc('xtick', labelsize='small')
    plt.rc('ytick', labelsize='small')
    plt.xlabel('Raman Shift (cm⁻¹)')
    plt.ylabel('Intensity')

    # Use the length of y_values_excluded as the length for x_values
    x_values = np.arange(134, 166)

    # Define the columns to exclude (134 to 162)
    excluded_columns = [str(i) for i in range(134, 166)]

    # Create y_values for the excluded region
    y_values_excluded = data[excluded_columns].iloc[33]

    # Plot the excluded region in red
    plt.plot(x_values, y_values_excluded, color='red')

    # first part
    y1 = [193464] * 34
    x1 = np.arange(100, 134)
    plt.plot(x1, y1, color='red')

    # third part
    y3 = [195635] * 34
    x3 = np.arange(166, 200)
    plt.plot(x3, y3, color='red')

    plt.ylim(180000, 220000)

    # Show legend
    plt.legend()

    # Save the plot to a file for inclusion in your paper
    plt.savefig('plots/justglutathione.png', dpi=300, bbox_inches='tight')  # Adjust the file format and resolution as needed


    plt.show()

    return None

def plot_spectra_simple(data):
    # Customize plot styles for better readability
    plt.rc('font', family='serif', size=12)
    plt.rc('xtick', labelsize='small')
    plt.rc('ytick', labelsize='small')
    plt.xlabel('Raman Shift (cm⁻¹)')
    plt.ylabel('Intensity')

    x_values = np.arange(0, len(data))
    y_values = data

    # Plot the entire spectrum in blue
    plt.plot(x_values, y_values, color='blue', label='BSA Signal')

    # Highlight the specified range in red
    plt.plot(x_values[140:161], y_values[140:161], color='red', label='Disulfide Signal')

    # Show legend
    plt.legend()

    # Save the plot to a file for inclusion in your paper
    plt.savefig('plots/bothcomponents.png', dpi=300, bbox_inches='tight')  # Adjust the file format and resolution as needed


    plt.show()

    return None

def simple_plot(df, df1, df2):
    # Define colors for each line
    colors = ['blue', 'green', 'red']

    x = np.arange(400, 2800, 10.29)


    # Plot each spectrum with a specific color
    plt.plot(np.arange(400, 2800, 7.73), df, color='0')
    plt.plot(np.arange(400, 2800, 7.73), df1, color='0')
    plt.plot(np.arange(400, 2800, 7.73), df2, color='0')

    plt.gcf().set_size_inches(8, 5)

    # Add labels and title
    plt.xlabel('Raman Shift (cm⁻¹)', fontsize = 12)
    plt.ylabel('Intensity', fontsize = 12)
    plt.title('Glutathione in PEG', fontsize = 12)

    # Customize ticks and labels
    plt.xticks(fontsize=12)  # X-axis tick font size
    plt.yticks(fontsize=12)  # Y-axis tick font size

    # Set x-axis ticks to display integers
    x_ticks_positions = np.arange(400, 2800, 300)
    x_ticks_labels = np.arange(400, 2800, 300) #[str(int(pos)) for pos in x_ticks_positions]
    plt.xticks(x_ticks_positions, x_ticks_labels, fontsize=12)

    # Add legend
    plt.legend()

    # Save figure
    #plt.savefig('plots/red_and_ox_in_PEG.png', dpi=1200, bbox_inches='tight', transparent=True)

    # Show the plot
    plt.show()

    return None

def simple_plot(df, df1, df2, df3):
    # Define colors for each line
    colors = ['blue', 'green', 'red', 'purple']

    x = np.arange(400, 2800, 10.29)

    # Plot each spectrum with a specific color and label
    plt.plot(np.arange(400, 2800, 7.73), df, color=colors[0], label='Water')
    plt.plot(np.arange(400, 2800, 7.73), df1, color=colors[1], label='Phosphate')
    plt.plot(np.arange(400, 2800, 7.73), df2, color=colors[2], label='PEG')
    plt.plot(np.arange(400, 2800, 7.73), df3, color=colors[3], label='Glutathione in PEG')

    plt.gcf().set_size_inches(8, 5)

    # Add labels and title
    plt.xlabel('Raman Shift (cm⁻¹)', fontsize=12)
    plt.ylabel('Intensity', fontsize=12)
    plt.title('Spectral Components', fontsize=12)

    # Customize ticks and labels
    plt.xticks(fontsize=12)  # X-axis tick font size
    plt.yticks(fontsize=12)  # Y-axis tick font size

    # Set x-axis ticks to display integers
    x_ticks_positions = np.arange(400, 2800, 300)
    x_ticks_labels = np.arange(400, 2800, 300)  # [str(int(pos)) for pos in x_ticks_positions]
    plt.xticks(x_ticks_positions, x_ticks_labels, fontsize=12)

    # Add legend
    plt.legend()

    # Save figure
    plt.savefig('plots/red_and_ox_in_PEG.png', dpi=1200, bbox_inches='tight', transparent=True)

    # Show the plot
    plt.show()

    return None

def plot_correlated_points(corr, spec):
    xpoints = []
    ypoints = []

    for i in range(len(corr)):
        if corr.iloc[i]['153'] > 0.7:
            xpoints.append(corr.iloc[i]['point'])
            point = corr.iloc[i]['point']
            ypoints.append(corr.iloc[0][int(point)])

    # Sort the x-points to ensure they are in ascending order
    sorted_indices = sorted(range(len(xpoints)), key=lambda k: xpoints[k])
    xpoints = [xpoints[i] for i in sorted_indices]
    ypoints = [ypoints[i] for i in sorted_indices]

    plt.plot(spec.iloc[0], label='Line Plot')
    plt.plot(xpoints, ypoints, 'ro', markersize=3, label='Points')
    plt.legend()
    plt.show()

def make_glu_peak_comparison_plot(BSA, PEG, phos):
    # Define colors for each line
    colors = ['blue', 'green', 'red']

    # Plot each spectrum with a specific color
    plt.plot(BSA, label='BSA', color=colors[0])
    plt.plot(PEG, label='PEG', color=colors[1])
    plt.plot(phos, label='phos', color=colors[2])

    plt.gcf().set_size_inches(10, 5)

    # Set x-axis limits to zoom in on the region from x = 145 to x = 160
    #plt.xlim(700, 800)
    #plt.ylim(-0.1, .5)

    # Add labels and title
    plt.xlabel('Raman Shift (cm⁻¹)', fontsize=12)
    plt.ylabel('Intensity', fontsize=12)
    plt.title('90 mM GSSG Peak Comparison', fontsize=12)

    # Customize ticks and labels
    plt.xticks(fontsize=12)  # X-axis tick font size
    plt.yticks(fontsize=12)  # Y-axis tick font size

    # Set x-axis ticks for the zoomed-in region
    # x_ticks_positions = np.arange(700, 800, 5)
    # x_ticks_labels = [str(int(pos)) for pos in x_ticks_positions]
    # plt.xticks(x_ticks_positions, x_ticks_labels, fontsize=12)

    plt.grid(True)

    # Add legend
    plt.legend()

    # Save figure
    #plt.savefig('plots/50mM_GSSG_peakcomparison.png', dpi=1200, bbox_inches='tight')

    # Show the plot
    plt.show()

    return None

if __name__ == '__main__':
    bsa = pd.read_csv('data/old data 2-16-2024/data_610.csv')
    peg = pd.read_csv('data/old data 2-16-2024/peg_prepro_610.csv')
    phos = pd.read_csv('data/old data 2-16-2024/phos_prepro_610.csv')

    make_glu_peak_comparison_plot(bsa.iloc[2], peg.iloc[4], phos.iloc[1])
    exit()
    # PCA1 vs PCA2
    data, ratio = PCA1(data2, 5)
    print(ratio)

    bsadf = separate_by_sol_andplot(data, soldata['BSA'])
    pegdf = separate_by_sol_andplot(data, soldata['PEG'])
    phosdf = separate_by_sol_andplot(data, soldata['phos'])
    plt.scatter(bsadf[0], bsadf[1], c='g')
    plt.scatter(pegdf[0], pegdf[1])
    plt.scatter(phosdf[0], phosdf[1])
    plt.show()

