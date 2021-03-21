# File name: PCA_sklearn

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn import datasets


def loading_score(pc, index):
    # Get the name of the top 10 samples that contribute most to pc1.

    # Get the loading scores
    loading_scores = pd.Series(data=pc.components_[0], index=index)
    # Sort the loading scores based on their magnitude
    sorted_loading_scores = loading_scores.abs().sort_values(ascending=False)
    # Get the names of the top 10
    top_10 = sorted_loading_scores[0:10].index.values
    # Print the names and their scores
    print(loading_scores[top_10])


def visualize_pc(df, ev_ratio):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Principal Component 1 - EVR: {0}%'.format(ev_ratio[0]), fontsize=15)
    ax.set_ylabel('Principal Component 2 - EVR: {0}%'.format(ev_ratio[1]), fontsize=15)
    ax.set_title('2 Component PCA', fontsize=20)

    targets = ['setosa', 'versicolor', 'virginica']
    colors = ['r', 'g', 'b']
    markers = ['x', '.', 'o']
    for target, color, marker in zip(targets, colors, markers):
        indices = df['flower name'] == target
        ax.scatter(df.loc[indices, 'PC1']
                   , df.loc[indices, 'PC2']
                   , c=color
                   , marker=marker
                   , s=50)
    ax.legend(targets)
    ax.grid()
    plt.show()


def explained_variance_ratio_plot(evr):

    labels = ['PC' + str(x) for x in range(1, len(evr) + 1)]

    plt.bar(x=range(1, len(evr) + 1), height=evr, tick_label=labels)
    plt.ylabel('Percentage of Explained Variance')
    plt.xlabel('Principal Component')
    plt.title('Screening PC importance')
    plt.grid()
    plt.show()


def pandas_df(iris_data):
    # Create a pandas data frame from iris data
    iris_df = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)
    # Append a target to the data frame
    iris_df['target'] = iris_data.target
    # Append a new column called flower
    iris_df['flower name'] = iris_df.target.apply(lambda x: iris_data.target_names[x])
    pd.set_option('display.max_columns', None)
    # Create X and y by dropping some columns from iris data frame
    xx = iris_df.drop(['target', 'flower name'], axis='columns')
    yy = iris_df['flower name']

    return xx, yy


if __name__ == '__main__':
    """ Principal Component Analysis using SKlearn"""
    # data = datasets.load_digits()
    data = datasets.load_iris()

    # Create a pandas data frame from iris data
    X, y = pandas_df(data)

    # Although, all features in the Iris dataset were measured in centimeters, let us continue
    # with the transformation of the data onto unit scale (mean=0 and variance=1),
    # which is a requirement for the optimal performance of many machine learning algorithms.
    X_scaled = StandardScaler().fit_transform(X)

    # PCA Projection to 2D
    pca = PCA()
    # Get PCA coordinates for scaled X data
    principal_components = pca.fit_transform(X_scaled)

    # Create a dataframe for the calculated principal components
    columns = ['PC' + str(i) for i in range(1, principal_components.shape[1] + 1)]
    df_pc = pd.DataFrame(data=principal_components,
                         columns=columns)

    # Add or concatenate df_pc with another dataframe (y)
    df_ = pd.concat([df_pc, y], axis=1)

    # Calculate the explained variance ratio
    explained_variance_ratio = np.round(pca.explained_variance_ratio_ * 100, decimals=1)
    # Plot the explained_variance_ratio - screening the PC importance
    explained_variance_ratio_plot(explained_variance_ratio)

    # Visualize the PCA
    visualize_pc(df_, explained_variance_ratio)

    # Determine which sample had the biggest influence on PC1
    loading_score(pca, X.columns)


