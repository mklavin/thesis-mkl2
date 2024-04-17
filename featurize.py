import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""
generate new features from integrated intensities of peaks!
"""

def integrate_every_five(df, dx):
    new_columns = []
    for i in range(0, len(df.columns), dx):
        cols = df.columns[i:i+dx]
        new_column_name = i//dx
        new_columns.append(new_column_name)
        df[new_column_name] = df[cols].sum(axis=1)
    return df[new_columns]

if __name__ == '__main__':
    GSH_df = pd.read_csv('data/raman_prepro_610.csv')

    integrate_every_five(GSH_df, 10).to_csv('data/integrated spectra/600gg_data_GSH_10.csv', index=False)

