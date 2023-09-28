import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math

def separate_bysol(df):
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
    folder_path = 'data/raw_data'  # filepath to folder of data
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
    data['conc_GSSG'] = data['names'].str.extract(r'(?:.*GSH.*)?(\d+\s*mM)')[0]  # these lines from chatGPT
    data['conc_GSSG'] = data['conc_GSSG'].str.replace('mM', '').str.strip()

    contains_580 = data[data['names'].str.contains('580')]
    contains_610 = data[data['names'].str.contains('610')]

    #contains_580 = contains_580.drop(columns=['names', 'conc_GSSG'])
    #contains_610 = contains_610.drop(columns=['names', 'conc_GSSG'])

    return contains_580, contains_610

def drop_missingvals(spectra):
    spectra = spectra.drop(columns='563')
    return spectra

def sort_usingsol_index(df, key):
    for i in df[key]:
        if is_nan_string(i) == True:
            result.to_csv('data/phosdata_580_BR_NM.csv', index=False)
            break
        i = int(i)
        row = contains_580.iloc[i]
        row = pd.DataFrame([row])
        result = pd.concat([result, row], ignore_index=True)

    return None

if __name__ == '__main__':
    df = pd.read_csv('data/data_610_names.csv')
    contains_580 = pd.read_csv('data/data_610_BR_NM.csv')

    result = pd.DataFrame(columns=contains_580.columns)
    separate_bysol(df).to_csv('data/separate_by_sol_610.csv')


    # continue trying to make dataframes separated by sample
    # why does it leave the last value out?
    # why doesn't BSA work?
    # then perform pca on different solutions- prove to casey which component is which




