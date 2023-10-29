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
from preprocessing import normalize, smooth_spectra

def polynomial_search(df, conc):
    func = [pybaselines.polynomial.poly, pybaselines.polynomial.modpoly, pybaselines.polynomial.imodpoly,
            pybaselines.polynomial.penalized_poly, pybaselines.polynomial.quant_reg,
            pybaselines.polynomial.goldindec]
    # pybaselines.polynomial.loess,
    results = []

    # polynomial fitting for baseline removal
    for j in func:
        polyorders = []
        for x in range(1, 10):
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

            # Add an index column
            # df.reset_index(inplace=True)

            # Sort column 'A' in ascending order while preserving the index
            sorted_df = baseline_removed.sort_values(by=1338, ascending=False)
            sorted_df = sorted_df[[1338]].iloc[1:]
            polyorders.append(max(sorted_df[1338]))
        max_poly = polyorders.index(max(polyorders))
        results.append([j, max_poly, max(polyorders)])

    # max_index = results.index(max(results))

    print(max(results, key=lambda x: x[2]))

    return None

def baseline_search(df, conc):
    func = [pybaselines.whittaker.asls, pybaselines.whittaker.iasls, pybaselines.whittaker.airpls,
            pybaselines.whittaker.arpls, pybaselines.whittaker.drpls, pybaselines.whittaker.iarpls,
            pybaselines.whittaker.aspls, pybaselines.whittaker.psalsa, pybaselines.whittaker.derpsalsa,
            pybaselines.morphological.mpls, pybaselines.morphological.mor, pybaselines.morphological.imor,
            pybaselines.morphological.mormol, pybaselines.morphological.amormol, pybaselines.morphological.rolling_ball,
            pybaselines.morphological.mwmv, pybaselines.morphological.tophat, pybaselines.morphological.mpspline,
            pybaselines.morphological.jbcd, pybaselines.spline.mixture_model, pybaselines.spline.irsqr,
            pybaselines.spline.corner_cutting, pybaselines.spline.pspline_asls, pybaselines.spline.pspline_iasls,
            pybaselines.spline.pspline_airpls, pybaselines.spline.pspline_arpls, pybaselines.spline.pspline_drpls,
            pybaselines.spline.pspline_iarpls, pybaselines.spline.pspline_aspls, pybaselines.spline.pspline_psalsa,
            pybaselines.spline.pspline_derpsalsa, pybaselines.smooth.noise_median, pybaselines.smooth.snip,
            pybaselines.smooth.swima, pybaselines.smooth.ipsa, pybaselines.smooth.ria]
    results = []

    # polynomial fitting for baseline removal
    for j in func:
        baseline_removed = []
        for index, rowy in df.iterrows():
            row = rowy.values.reshape(-1, 1)
            row = row.flatten()
            row_polyfit = j(row)[0]
            # plt.title(str(i))
            # plt.plot(row)
            # plt.plot(row_polyfit)
            # plt.show()
            row = row - row_polyfit
            row = row.flatten()
            row = row.reshape(1, -1)
            normalized_df = pd.DataFrame(row, columns=rowy.index)
            baseline_removed.append(normalized_df)

        baseline_removed = pd.concat(baseline_removed, axis=0, ignore_index=True)
        #baseline_removed = smooth_spectra(baseline_removed)
        baseline_removed = normalize(baseline_removed)
        baseline_removed = pd.concat([baseline_removed, conc], axis=1)
        baseline_removed = pd.DataFrame(np.corrcoef(baseline_removed.T))

        # Add an index column
        # df.reset_index(inplace=True)

        # Sort column 'A' in ascending order while preserving the index
        sorted_df = baseline_removed.sort_values(by=1338, ascending=False)
        sorted_df = sorted_df[[1338]].iloc[1:]
        results.append(max(sorted_df[1338]))

    max_index = results.index(max(results))

    print(str(func[max_index]), max(results))

    return None



if __name__ == '__main__':
    df = pd.read_csv('data/data_610.csv')
    conc = pd.read_csv('data/data_610_concentrations_GSH.csv')


    polynomial_search(df, conc)