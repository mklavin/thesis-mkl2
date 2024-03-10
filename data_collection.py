import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
from preprocessing import PCA1
from sklearn import decomposition

# functions used for organizing raw Raman data files

def combine_data():
    # used to combine daniels raman data and my raman data
    daniels580 = pd.read_csv('data/12_20_2023_newdata/new_12_20_data_580.csv')
    daniels610 = pd.read_csv('data/12_20_2023_newdata/new_12_20_data_610.csv')

    data580 = pd.read_csv('data/old data 2-16-2024/data_580.csv')
    data610 = pd.read_csv('data/old data 2-16-2024/data_610.csv')

    conc610 = pd.read_csv('data/old data 2-16-2024/data_610_concentrations_GSH.csv')
    conc580 = pd.read_csv('data/old data 2-16-2024/data_580_concentrations_GSSG.csv')

    names580 = pd.read_csv('data/old data 2-16-2024/data_580_names.csv')
    names610 = pd.read_csv('data/old data 2-16-2024/data_610_names.csv')

    conc610 = pd.concat([conc610['conc_GSH'], daniels610['conc_GSH']])
    conc580 = pd.concat([conc580['conc_GSSG'], daniels580['conc_GSSG']])

    names610 = pd.concat([names610['names'], daniels610['names']])
    names580 = pd.concat([names580['names'], daniels580['names']])

    conc580.to_csv('data/raman_580_concentrations_GSSG.csv', index=False)
    conc610.to_csv('data/raman_610_concentrations_GSH.csv', index=False)
    names580.to_csv('data/raman_580_names.csv', index=False)
    names610.to_csv('data/raman_610_names.csv', index=False)

    daniels610 = daniels610.drop(columns=['conc_GSH', 'names', '563'])
    daniels580 = daniels580.drop(columns=['conc_GSSG', 'names', '563'])
    daniels610 = pd.concat([data610, daniels610])
    daniels580 = pd.concat([data580, daniels580])
    daniels580.to_csv('data/raman_580.csv', index=False)
    daniels610.to_csv('data/raman_610.csv', index=False)

    return None

def separate_bysol(df):
    """
    used to find the indices of each solvent
    :param df: dataframe with a column of spectral names
    :return: dataframe with columns named by solvents where rows are indices of spectra
    of a particular solvent from the input df
    """

    # Define the substrings to search for
    substrings = ['BSA', 'PEG', 'phos']

    # Initialize a dictionary to store the column values
    result_dict = {substring: [] for substring in substrings}
    result_dict['phos'] = []

    # Iterate through the DataFrame and check for each substring
    for index, row in df.iterrows():
        found_substring = False

        for substring in substrings:
            if substring in row['names']:
                result_dict[substring].append(index)
                found_substring = True

        if not found_substring:
            result_dict['phos'].append(index)

    # Find the maximum length of the lists in the dictionary
    max_length = max(len(indices) for indices in result_dict.values())

    # Pad the lists with NaN values to make them the same length
    for key in result_dict.keys():
        result_dict[key].extend([np.nan] * (max_length - len(result_dict[key])))

    # Convert the dictionary to a DataFrame
    result_df = pd.DataFrame(result_dict)

    # result_df.to_csv('data/separate_by_sol_580.csv', index=False)
    return result_df


def is_nan_string(string):
    try:
        # Attempt to convert the string to a float
        value = float(string)
        # Check if the float is NaN
        return math.isnan(value)
    except ValueError:
        # If the conversion raises a ValueError, it's not a valid number
        return False

def stitch_spectra(spec_580, spec_610):
    """
    used to stitch spectral windows
    :param spec_580: dataframe containing 580 spectra
    :param spec_610: dataframe containing 610 spectra
    :return: dataframe of combined spectra along the x-axis
    """

    # Initialize an empty list to store DataFrames for concatenation
    combined_dfs = []

    # Drop the 'names' column from both DataFrames
    spec_580 = spec_580.drop(columns=['names'])
    spec_610 = spec_610.drop(columns=['names'])

    # Loop through each row in spec_580
    for index, row_580 in spec_580.iterrows():
        # Extract the concentration value from spec_580
        conc_580 = row_580['conc_GSSG']
        if is_nan_string(conc_580) == True:
            conc_580 = 0
        conc_580 = int(conc_580)

        # Get the first 418 values from spec_580
        row_580_values = row_580.iloc[:418].tolist()

        # Loop through each row in spec_610
        for _, row_610 in spec_610.iterrows():
            # Extract the concentration value from spec_610
            conc_610 = row_610['conc_GSSG']
            print(conc_610)
            if is_nan_string(conc_610) == True:
                conc_610 = 0
            conc_610 = int(conc_610)

            # Get the first 418 values from spec_610
            row_610_values = row_610.iloc[:418].tolist()

            if conc_580 == 0:
                conc_580 = .01
            if conc_610 == 0:
                conc_610 = .01

            # Combine the data from both rows along with the concentration ratio
            combined_data = row_580_values + row_610_values + [conc_580 / conc_610]

            # Create a DataFrame from the combined data
            combined_df = pd.DataFrame([combined_data])

            # Append the DataFrame to the list for later concatenation
            combined_dfs.append(combined_df)

    # Concatenate all DataFrames in the list into a single DataFrame
    newdata = pd.concat(combined_dfs, ignore_index=True)

    return newdata

def gather_data():
    # loops through all of the raw data files in data and converts to one csv file
    # also makes files with the name of data and labels

    folder_path = 'data/March 7 Data Collection'  # filepath to folder of data
    n = 1340  # length of spectra
    columns = [i for i in range(1, n)]  # create list of same length
    data = pd.DataFrame(columns=columns)  # empty dataframe with columns for each value
    names = []  # store names of files
    for file in os.listdir(folder_path):
        # loop through files in folder
        filepath = os.path.join(folder_path, file)
        text = pd.read_csv(filepath, names=['x', 'col2', 'y'])  # import text file
        names.append(file)
        data.loc[len(data.index)] = text['y'].transpose()  # add text file data to the dataframe
    data.insert(0, 'names', names, True)  # add name column

    # adding feature columns and labels
    data['conc_GSH'] = data['names'].str.extract(r'(?:.*GSSG.*)?(\d+\s*mM)')[0]  # these lines from chatGPT
    data['conc_GSH'] = data['conc_GSH'].str.replace('mM', '').str.strip()

    # adding feature columns and labels
    data['conc_GSSG'] = data['names'].str.extract(r'(?:.*GSH.*)?(\d+\s*mM)')[0]  # these lines from chatGPT
    data['conc_GSSG'] = data['conc_GSSG'].str.replace('mM', '').str.strip()

    # Reorder the columns to have concentration as the two columns
    data = data[['conc_GSH'] + [col for col in data.columns if col != 'conc_GSH']]
    data = data[['conc_GSSG'] + [col for col in data.columns if col != 'conc_GSSG']]

    contains_580 = data[data['names'].str.contains('150gg')] #data[data['names'].str.contains('580')]
    contains_610 = data[data['names'].str.contains('610')]

    #contains_580 = contains_580.drop(columns=['conc_GSH', 563])
    #contains_610 = contains_610.drop(columns=['names', 'conc_GSSG'])

    return contains_580#, contains_610

def drop_missingvals(spectra):
    # raman spectrometer is missing a pixel at the 563rd position
    spectra = spectra.drop(columns='563')
    return spectra

def is_nan_string(value):
    return value != value  # Check if the value is NaN

def sort_usingsol_index(df, df2, key):
    """
    :param df: dataframe of all data
    :param df2: dataframe containing indices of different solvents
    :param key: which solvent to search for
    :return: dataframe of samples in specified solvent
    """
    result = pd.DataFrame()
    for i in df[key]:
        if is_nan_string(i):
            continue  # Skip NaN values
        i = int(i)
        if i >= len(df2):
            continue  # Skip if the index is out of bounds
        row = df2.iloc[i]
        row = pd.DataFrame([row])
        result = pd.concat([result, row], ignore_index=True)
    return result

def cut_spectra(df, region=str):
    if region == '580':
        start = 26
        end = 213
    if region == '610':
        start = 669
        end = 834

    return df.iloc[:, start:end]

def select_corr_points(df, corr):
    # used to select for correlated points of a certain value

    # Initialize an empty list to store DataFrames
    selected_rows = []

    # Transpose the original DataFrame
    df = df.T

    # Remove rows where values in column 'A' are less than 0.5
    corr = corr[corr['1338'] >= 0.5]

    # Loop through the indices in 'corr'
    for i in corr['index']:
        # Check if the index is within the valid range of 'df'
        if i < len(df):
            row = df.iloc[i]
            selected_rows.append(row)

    # Concatenate the selected rows into a new DataFrame
    final_df = pd.concat(selected_rows, axis=1).T

    return final_df.T

def get_concentration_BSA(absp, pathlength):
    # beers law, use this function during sample prep to get the correct concentration of BSA (0.6 mM)
    x = absp/(43824*pathlength)*1000
    print('concentration of BSA:', x, 'mM')
    return None

def remove_pixel(df):
    df = df.drop(columns=['563'], axis=1)
    df.to_csv('data/new_data_610.csv', index=False)
    return None

def check_rows_in_dataframe(dataframe1, dataframe2): # written by chatGPT
    # Convert DataFrames to sets for efficient comparison
    set_dataframe1 = set(map(tuple, dataframe1.values))
    set_dataframe2 = set(map(tuple, dataframe2.values))

    # Check if any rows of dataframe1 are in dataframe2
    rows_in_dataframe2 = set_dataframe1.intersection(set_dataframe2)

    return len(rows_in_dataframe2)

def reorder_rows(df, indices):
    newdf = df.copy()

    for i in indices:
        newdf.iloc[i] = df.iloc[i][::-1].values

    return newdf

if __name__ == '__main__':
    df = pd.read_csv('data/March 7 Data Collection/150gg_data.csv')
    df2 = pd.read_csv('data/names_150gg_data.csv')

    excluded_values = [64, 65, 66, 73, 74, 75, 76, 77, 78, 79, 80, 82, 84, 86, 88, 89, 90, 91, 93, 95, 96, 99, 101, 102, 103, 104, 105, 106, 110, 111]

    generated_list = [x for x in range(113) if x not in excluded_values]
    print(generated_list)

    df = reorder_rows(df, generated_list)
    print(df)


    # now the rows are flipped
    # need to delete the bad rows, then zoom in

    df = df.iloc[:, 523:882]
    df = df.drop(columns= ['564'])
    for i in range(len(df)):
        plt.plot(df.iloc[i])
        plt.title(str(df2.iloc[i]))
        plt.show()

    # spectra are weirdly not aligned??
    # check how the flipping worked


    exit()
    for i in range(len(df)):
        plt.plot(df.iloc[i])
        plt.title(str(df2.iloc[i]))
        plt.show()
















