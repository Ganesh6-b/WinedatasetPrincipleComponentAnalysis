# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 18:42:35 2019

@author: Ganesh
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset = pd.read_csv("F:\\R\\files\\wine.csv")

pca_dataset = dataset.iloc[:,1:]

#normalizing the dataset

from sklearn import preprocessing

pca_dataset_norm = pca_dataset

pca_dataset_norm = preprocessing.normalize(pca_dataset_norm)
#preparing pca
from sklearn.decomposition import PCA
pca = PCA()
pca_values = pca.fit_transform(pca_dataset_norm)

pca_values.shape
#visualizing
import seaborn as sns

sns.heatmap(pca_values)

#variance

var = pca.explained_variance_ratio_
var

#cumulative variance
var1 = np.cumsum(np.round(var, decimals = 4)*100)
var1

#plot between pca1 and pca2

x = np.array(pca_values[:,0])
y = np.array(pca_values[:,1])

plt.plot(x,y, "ro")
np.corrcoef(x,y)

#new dataframe

new_df = pd.DataFrame(pca_values[:,0:4])

from sklearn.cluster import KMeans

cluster1 = KMeans(n_clusters = 3).fit(new_df)

cluster1.labels_

new_labels = np.array(cluster1.labels_)

#find aggregate
new_df.groupby(new_labels).mean()
