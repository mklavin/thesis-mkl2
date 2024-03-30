import pandas as pd
from visualize import plot_and_cluster_kmeans, sort_clustering_labels, plot_data_with_colors
import matplotlib as plt


if __name__ == '__main__':
    df = pd.read_csv('data/correlation analysis/150gg_data_scaled_only.csv')
    df2 = pd.read_csv('data/150gg_data_prepro.csv')

    labels = plot_and_cluster_kmeans(df, 5)
    sorted = sort_clustering_labels(labels)

    plot_data_with_colors(sorted, df2.iloc[56])

