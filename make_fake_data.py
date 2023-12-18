import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# fake data generation is a work in progress :)

def get_slope_and_intercept(df1, conc):
    # this function finds the slope and intercept for the line of best fit
    # between the most correlated points (corrcoef > 0.75) and the labels
    # this can be used to create data later where points are updated based on the line of best fit
    # returns a dataframe with correlated point and corresponding slope and intercept

    df = pd.concat([df1, conc], axis=1)
    df = pd.DataFrame(np.corrcoef(df.T))
    df = df.sort_values(by=1338, ascending=True) # finding correlation coefficient

    # Select rows where values are greater than 0.75
    selected_rows = df[df[1338] > 0.75]

    # Get a list of indices for the selected rows
    selected_indices = selected_rows.index.tolist()

    # Lists to store slopes and intercepts
    slopes = []
    intercepts = []

    for i in range(len(selected_indices)):
        x = list(df1[str(selected_indices[i])])
        y = list(conc['conc_GSSG'])

        # Fit a linear regression model
        slope, intercept = np.polyfit(x, y, 1)

        # Append slopes and intercepts to the respective lists
        slopes.append(slope)
        intercepts.append(intercept)

    # Create a DataFrame from the lists
    result_df = pd.DataFrame({'Point': selected_indices,'Slope': slopes, 'Intercept': intercepts})

    return result_df

def update_points(df, bestfit):
    # uses the line of best fit to generate new spectra
    # changes the points based on new concentration
    newdata = []
    concs = []

    j = 0
    while j < 10:
        for spectra in range(len(df)):
            conc = np.random.randint(1, 99)
            newrow = df.iloc[spectra].copy()  # Copy the original row to avoid modifying the original DataFrame

            for i in range(len(bestfit)):
                point_column = bestfit['Point'].iloc[i]
                slope = bestfit['Slope'].iloc[i]
                intercept = bestfit['Intercept'].iloc[i]

                new_value = (conc - intercept) / slope # from line of best fit
                newrow[point_column] = new_value # new intensity = conc - int/slope
            newdata.append(newrow)
            concs.append(conc)
        j += 1

    updated_df = pd.concat(newdata, axis=1, ignore_index=True)
    concs = pd.DataFrame(concs, columns=['concs'])
    return updated_df.T, concs # returns new spectra and corresponding concentrations

if __name__ == '__main__':
    data = pd.read_csv('data/phos_prepro_580.csv')
    conc = pd.read_csv('data/phos_data_580_concentrations.csv')
    bestfit = pd.read_csv('data/fake_data/phos_data_580_lineofbestfit.csv')

    newdata, concs = update_points(data, bestfit)

    plt.plot(newdata.iloc[0])
    plt.show()



