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
    # find indices of each solution
    # used for analysis and semi-random training and test split
    # df contains spectra names 

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
    # combine two spectral windows
    # will use later

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

    folder_path = 'data/150 gg data'  # filepath to folder of data
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
    # Reorder the columns to have 'conc_GSH' as the first column
    data = data[['conc_GSH'] + [col for col in data.columns if col != 'conc_GSH']]

    contains_580 = data[data['names'].str.contains('580')]
    contains_610 = data[data['names'].str.contains('610')]

    contains_580 = contains_580.drop(columns=['conc_GSH', 563])
    #contains_610 = contains_610.drop(columns=['names', 'conc_GSSG'])

    return contains_580#, contains_610

def drop_missingvals(spectra):
    # raman spectrometer is missing a pixel at the 563rd position
    spectra = spectra.drop(columns='563')
    return spectra

def sort_usingsol_index(df, df2, key):
    """
    :param df: dataframe of all data
    :param df2: dataframe containing indices of different solvents
    :param key: which solvent to search for
    :return: dataframe of samples in specified solvent
    """
    result = pd.DataFrame()
    for i in df[key]:
        if is_nan_string(i) == True:
            result.to_csv('data/phos_data_580.csv', index=False)
            break
        i = int(i)
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

if __name__ == '__main__':
    df = pd.read_csv('data/raman_580_names.csv')
    df2 = pd.read_csv('data/raman_580.csv')

    # testy = {'1': [1, 2, 3], '2': [2, 4, 5.5], '3': [2, -1, 7], '4': [-2, 1, -7]}
    # testy2 = {'1': [1, 2, 3], '2': [2, 4, 5.5], '3': [3.9, 8, 9.1]}
    # testy = pd.DataFrame(data=testy)
    # testy2 = pd.DataFrame(data=testy2)
    # print(testy.T)
    #
    # # apply PCA
    # pca = decomposition.PCA(n_components=1)
    # X = pca.fit_transform(testy.T)
    # loadings = pd.DataFrame(pca.components_.T, columns=['PC1'])
    # print(loadings)
    # exit()

    separate_bysol(df).to_csv('data/separate_by_sol_580.csv', index=False)


    exit()

    df = pd.read_csv('data/150 gg data/150ggdata_cut.csv')
    vals = list(df['685'])
    for i in range(len(vals)):
        df.iloc[i] = df.iloc[i]-(vals[i]-95719)
    plt.plot(df.iloc[0])
    plt.plot(df.iloc[1])
    plt.plot(df.iloc[2])
    plt.plot(df.iloc[3])
    plt.plot(df.iloc[4])
    plt.show()
    exit()
    # 519-841

    data_580 = pd.read_csv('data/12_20_2023_newdata/new_12_20_data_580.csv')
    data_610 = pd.read_csv('data/12_20_2023_newdata/new_12_20_data_610.csv')
    data1 = pd.read_csv('data/old data 2-16-2024/separate_by_sol_580.csv')

    combine_data()
    exit()

    conc_580 = data_580['conc_GSSG']
    conc_610 = data_610['conc_GSH']

    names_580 = data_580['names']
    names_610 = data_610['names']

    conc_580.to_csv('data/new_data_580_concentrations_GSSG.csv', index=False)
    conc_610.to_csv('data/new_data_610_concentrations_GSH.csv', index=False)
    names_580.to_csv('data/new_data_580_names.csv', index=False)
    names_610.to_csv('data/new_data_610_names.csv', index=False)

    data_580 = data_580.drop(columns=['conc_GSSG', 'names'])
    data_610 = data_610.drop(columns=['conc_GSH', 'names'])

    data_580.to_csv('data/new_data_580.csv', index=False)
    data_610.to_csv('data/new_data_610.csv', index=False)



    exit()

    x = sort_usingsol_index(data1, data, 'phos')

    x.to_csv('data/phos_data_580.csv', index=False)









