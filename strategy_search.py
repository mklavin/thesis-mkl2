import matplotlib.pyplot as plt
import pandas as pd
import pybaselines.whittaker
import pybaselines.morphological
import pybaselines.spline
import pybaselines.polynomial
import pybaselines.smooth
import pybaselines.classification
import pybaselines.optimizers
import pybaselines.misc
import numpy as np
from preprocessing import remove_baseline, normalize, smooth_spectra, scale_rows_to_max
from itertools import permutations

def put_together_preprocess_search(data, conc, region:str):
    """
    :param data: dataframe of raw raman data
    :param conc: concentrations of each spectra
    :param region: region of the raman spectra
    :return: preprocessed dataframe that gives the strongest correlation

    search algorithm for the best preprocessing method
    combining baseline removal, smoothing, normalization, and standardization
    uses correlation analysis to evaluate
    warning: takes a long time to run!
    """
    options = ['normalizing', 'baseline removal', 'standardize'] # possible preprocessing
    #  UNCOVER IF U WANT TO DO CHOOSE 1 OR CHOOSE 2
    # len_permutations = [2, 3]
    # total_perm = []
    # for i in len_permutations:
    #     all_permutations = list(permutations(options, i))
    #     total_perm.append(all_permutations)
    # total_perm = [item for sublist in total_perm for item in sublist] # making a bigger list of permutations
    total_perm = list(permutations(options, 3))
    total_perm = [['standardize', 'normalize']]
    results = []
    fits = []

    for n in total_perm: # for each permutation, find the correlation between the peak and concentration
        df = data
        for i in n: # preprocess!
            if i == 'normalizing':
                df = normalize(df)
            if i == 'standardize':
                df = scale_rows_to_max(df, region)
            if i == 'baseline removal':
                df, fit = baseline_search(df, conc, region)
                fits.append(fit)

        df = pd.concat([df, conc], axis=1)
        df = pd.DataFrame(np.corrcoef(df.T)) # CALCULATING COVARIANCE MATRIX

        # UNCOVER FOR 150GG DATA
        if region == '610':
            df = df.iloc[250:350]
        if region == '580':
            df = df.iloc[12:36]

        # UNCOVER FOR 600GG DATA
        # if region == '610':
        #     df = df.iloc[730:851]
        # if region == '580':
        #     df = df.iloc[140:160]

        sorted_df = df.sort_values(by=359, ascending=False) # only select the column that corresponds to concentrations
        sorted_df = sorted_df[[359]].iloc[1:]
        print(sorted_df)
        results.append(max(sorted_df[359]))
        print(max(sorted_df[359]), results)

    max_index = results.index(max(results))

    print(str(total_perm[max_index]), max(results), fits, max_index) # the best model with the best correlation

    df = data
    for i in total_perm[max_index]: # now we know the best preprocessing method. do this!
        if i == 'normalizing':
            df = normalize(df)
        if i == 'standardize':
            df = scale_rows_to_max(df, region)
        if i == 'baseline removal':
            df = remove_baseline(df, fits[max_index])

    return df # preprocessed dataframe

def polynomial_search(df, conc):
    # goes through each polynomial method finding the one with the best correlation
    func = [pybaselines.polynomial.poly, pybaselines.polynomial.modpoly, pybaselines.polynomial.imodpoly,
            pybaselines.polynomial.penalized_poly, pybaselines.polynomial.quant_reg,
            pybaselines.polynomial.goldindec]
    results = []

    # polynomial fitting for baseline removal
    for j in func:
        polyorders = []
        for x in range(3, 10):
            baseline_removed = []
            for index, rowy in df.iterrows():
                row = rowy.values.reshape(-1, 1)
                row = row.flatten()
                row_polyfit = j(row, poly_order=x)[0]
                row = row - row_polyfit
                row = row.flatten()
                row = row.reshape(1, -1)
                normalized_df = pd.DataFrame(row, columns=rowy.index)
                baseline_removed.append(normalized_df)

            baseline_removed = pd.concat(baseline_removed, axis=0, ignore_index=True)
            baseline_removed = smooth_spectra(baseline_removed)
            baseline_removed = normalize(baseline_removed)
            baseline_removed = pd.concat([baseline_removed, conc], axis=1)
            baseline_removed = pd.DataFrame(np.corrcoef(baseline_removed.T))

            # use 1338 for 600 gg data, 359 for 150 gg data
            sorted_df = baseline_removed.sort_values(by=359, ascending=False)
            sorted_df = sorted_df[[359]].iloc[1:]
            polyorders.append(max(sorted_df[359]))
        max_poly = polyorders.index(max(polyorders))
        results.append([j, max_poly, max(polyorders)])

    max_index = max(range(len(results)), key=lambda i: results[i][2])

    print(max(results, key=lambda x: x[2]))

    return remove_baseline(df, func[max_index], results[max_index][1]), func[max_index]

def baseline_search(df, conc, region):
    """
    :param df: dataframe of all raman spectra
    :param conc: concentrations of glutathione in each spectra
    :return: return the dataframe with the baseline subtracted that gives the highest correlation
    This function searches all of the pybaseline functions to find the one that gives the highest correlation
    between the glutathione peak and concentration
    """
    func = [pybaselines.whittaker.asls, pybaselines.whittaker.iasls, pybaselines.whittaker.airpls,
            pybaselines.whittaker.arpls, pybaselines.whittaker.drpls,
            pybaselines.whittaker.aspls, pybaselines.whittaker.psalsa, pybaselines.whittaker.derpsalsa,
            pybaselines.morphological.mpls, pybaselines.morphological.mor, pybaselines.morphological.imor,
            pybaselines.morphological.mormol, pybaselines.morphological.amormol, pybaselines.morphological.rolling_ball,
            pybaselines.morphological.mwmv, pybaselines.morphological.tophat, pybaselines.morphological.mpspline,
            pybaselines.morphological.jbcd, pybaselines.spline.mixture_model, pybaselines.spline.irsqr,
            pybaselines.spline.corner_cutting, pybaselines.spline.pspline_asls, pybaselines.spline.pspline_iasls,
            pybaselines.spline.pspline_airpls, pybaselines.spline.pspline_arpls, pybaselines.spline.pspline_drpls,
            pybaselines.spline.pspline_iarpls, pybaselines.spline.pspline_aspls, pybaselines.spline.pspline_psalsa,
            pybaselines.smooth.noise_median, pybaselines.smooth.snip,pybaselines.smooth.ipsa, pybaselines.smooth.ria, pybaselines.classification.dietrich,
            pybaselines.classification.golotvin, pybaselines.classification.std_distribution, pybaselines.classification.fastchrom, pybaselines.classification.fabc,
            pybaselines.optimizers.optimize_extended_range, pybaselines.optimizers.adaptive_minmax, pybaselines.misc.beads,
            pybaselines.smooth.swima]
    func = [pybaselines.spline.irsqr]
    results = []

    # polynomial fitting for baseline removal
    for j in func:
        baseline_removed = []
        for index, rowy in df.iterrows():
            row = rowy.values.reshape(-1, 1)
            row = row.flatten()
            row_polyfit = j(row)[0]
            # plt.plot(row)
            # plt.plot(row_polyfit)
            # plt.show()
            row = row - row_polyfit
            row = row.flatten()
            row = row.reshape(1, -1)
            normalized_df = pd.DataFrame(row, columns=rowy.index)
            baseline_removed.append(normalized_df)
        baseline_removed = pd.concat(baseline_removed, axis=0, ignore_index=True)
        baseline_removed = pd.concat([baseline_removed, conc], axis=1)
        baseline_removed = pd.DataFrame(np.corrcoef(baseline_removed.T))

        # UNCOVER FOR 150GG DATA
        if region == '610':
            baseline_removed = baseline_removed.iloc[250:350]
        if region == '580':
            baseline_removed = baseline_removed.iloc[12:36]

        # UNCOVER FOR 600GG DATA
        # if region == '610':
        #     baseline_removed = baseline_removed.iloc[730:851]
        # if region == '580':
        #     baseline_removed = baseline_removed.iloc[140:160]

        # Add an index column
        # baseline_removed.reset_index(inplace=True)

        # Sort column 'A' in ascending order while preserving the index
        sorted_df = baseline_removed.sort_values(by=359, ascending=False) # change to 1338 for 600 gg data, 359 for 150 gg data
        sorted_df = sorted_df[[359]].iloc[1:]
        results.append(max(sorted_df[359]))

    max_index = results.index(max(results))

    return remove_baseline(df, func[max_index]), func[max_index]



if __name__ == '__main__':
    df = pd.read_csv('data/apr13_data.csv')
    df2 = pd.read_csv('data/150gg_data_cropped.csv')
    conc = pd.read_csv('data/apr13_conc_GSSG.csv')
    conc2 = pd.read_csv('data/GSH_conc_150gg_data.csv')

    # for i in range(len(df)):
    #     plt.plot(df.iloc[i])
    #     plt.show()

    df = put_together_preprocess_search(df, conc, '610')
    # plt.plot(df.iloc[4])
    # plt.plot(df.iloc[6])
    # plt.plot(df.iloc[33])
    # plt.plot(df.iloc[22])
    # plt.show()
    df.to_csv('data/apr13_data_normalize_standardize.csv', index=False)


