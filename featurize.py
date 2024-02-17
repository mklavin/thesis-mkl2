import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from preprocessing import calc_correlation_matrix

def integrate_dataframe(df, increment_size=10):
    integrated_df = pd.DataFrame()

    # Iterate over each row in the DataFrame
    for index, row in df.iterrows():
        y_values = row.values  # Extract y values from the row

        # Calculate the number of increments
        num_increments = len(y_values) // increment_size

        # Initialize a new vector for the integrated values
        integrated_values = np.zeros(num_increments)

        # Integrate each increment of specified size
        for i in range(num_increments):
            start_index = i * increment_size
            end_index = (i + 1) * increment_size
            integrated_values[i] = np.trapz(y_values[start_index:end_index])

        # Create a new DataFrame with the integrated values for the current row
        integrated_df = pd.concat([integrated_df, pd.DataFrame(integrated_values).transpose()], ignore_index=True)

    return integrated_df

if __name__ == '__main__':
    data = pd.read_csv('data/old data 2-16-2024/prepro_580.csv')
    conc = pd.read_csv('data/old data 2-16-2024/data_580_concentrations_GSSG.csv')

    integrate_dataframe(data).to_csv('data/integrated_prepro_580.csv', index=False)


