import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
from preprocessing import PCA1
from sklearn import decomposition

"""
functions used for organizing raw Raman data files
very useful and more efficient than manual data organization
"""

def separate_by_solvent(df):
    """
    used to find the indices of each solvent based on spectral file names
    :param df: dataframe with a column of spectral names
    :return: dataframe with columns named by solvents where rows are indices of spectra
    of a particular solvent from the input df
    """

    # Define the substrings to search for
    substrings = ['BSA', 'PEG', 'phos'] # can change this!

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

    # result_df.to_csv('data/separate_by_sol_580.csv', index=False) # uncover to save to csv
    return result_df


def is_nan_string(string):
    # helper function
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

def gather_data(folder_path): # very useful function :)
    """
    when given a path to a folder containing raw data,
    makes a dataframe containing all of the spectra as rows.
    also adds columns for concentrations and file names.
    the last few lines are modified for 150gg data, but can be
    easily adjusted for 580 and 610 data.
    :param folder_path: string specifying directory to pull files
    :return: dataframe with raw spectral data in addition to name and concentration columns
    """

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

    contains_580 = data[data['names'].str.contains('150gg')] #data[data['names'].str.contains('580')] # replace with this for 580 data
    contains_610 = data[data['names'].str.contains('610')]

    #contains_580 = contains_580.drop(columns=['conc_GSH', 563]) # remove pixel 563
    #contains_610 = contains_610.drop(columns=['names', 'conc_GSSG'])

    return contains_580#, contains_610

def drop_missingvals(df):
    """
    our raman spec is missing a pixel at the 563rd position.
    drop that column!
    :param spectra: df containing all spectra
    :return: same dataframe, except col 563 is dropped
    """
    df = df.drop(columns='563')
    return df

def is_nan_string(value):
    # helper function
    return value != value  # Check if the value is NaN

def sort_usingsol_index(data, separate_by_sol, key):
    """
    generates dataframes with specified solvents only
    use the sepearate_by_solvent() function!
    :param data: dataframe of all data
    :param separate_by_sol: dataframe containing indices of different solvents
    this dataframe can be generated by the separate_by_solvent() function!
    :param key: which solvent to search for
    :return: dataframe of samples in specified solvent
    """
    result = pd.DataFrame()
    for i in data[key]:
        if is_nan_string(i):
            continue  # Skip NaN values
        i = int(i)
        if i >= len(separate_by_sol):
            continue  # Skip if the index is out of bounds
        row = separate_by_sol.iloc[i]
        row = pd.DataFrame([row])
        result = pd.concat([result, row], ignore_index=True)

    return result # dataframe of a particular solvent

def select_corr_points(df, corr, n):
    """
    useful for correlation analysis. can give columns above a certain threshold (n) for
    covariance to concentration.
    :param df: dataframe containing spectra
    :param corr: covaraince matrix
    :param n: covariance threshold to sort by (ex: 0.5)
    :return: dataframe containing spectral columns that are a specified covariance to concentration
    """

    # Initialize an empty list to store DataFrames
    selected_rows = []

    # Transpose the original DataFrame
    df = df.T

    # Remove rows where values in column 'A' are less than
    corr = corr[corr['359'] >= .7]  # change this if you want to do a different column
    corr = corr.sort_values(by='359')

    # Loop through the indices in 'corr'
    for i in corr['index']:
        # Check if the index is within the valid range of 'df'
        if int(i)-524 < len(df):
            row = df.iloc[int(i)-524]
            selected_rows.append(row)

    # Concatenate the selected rows into a new DataFrame
    final_df = pd.concat(selected_rows, axis=1).T

    return corr # can also change this to: final_df.T to get columns of a certain correlation

def reorder_rows(df, indices):
    """
    if raman spectra are flipped, use this function
    :param df: dataframe of spectral data
    :param indices: row indices to flip
    :return: dataframe of spectral data with specified columns flipped
    """
    newdf = df.copy()

    for i in indices:
        newdf.iloc[i] = df.iloc[i][::-1].values

    return newdf

if __name__ == '__main__':
    df = pd.read_csv('data/150gg_data_prepro.csv')
    corr = pd.read_csv('data/correlation analysis/150gg_data_prepro_GSH.csv')

    corr = pd.concat([corr['359'], pd.DataFrame(df.columns, columns=['index'])], axis=1)

    corr = corr.drop([359])
    print(corr)

    print(select_corr_points(df, corr, .8))
















