import warnings
warnings.simplefilter('ignore')

# ml-source_29-30_s1.py

# Grouping objects by similarity using k-means
## K-means clustering using scikit-learn

from sklearn.datasets import make_blobs

X, y = make_blobs(n_samples=150,   # total number of data points
                  n_features=2,    # number of features
                  centers=3,       # number of clusters
                  cluster_std=0.5, # within-cluster standard deviation
                  shuffle=True,    # shuffle data points
                  random_state=0)  # specify random number generator state
print(y)

import matplotlib.pyplot as plt

plt.scatter(X[:, 0], X[:, 1], 
            c='white', marker='o', edgecolor='black', s=50)
plt.grid()
plt.tight_layout()

# plt.savefig('images/simple_2d_dataset.png', dpi=300)
plt.show()

# applying a simple k-means mechanism to a sample dataset
from sklearn.cluster import KMeans

km = KMeans(n_clusters=3,   # number of clusters
            init='random',  # randomly select initial centroid values
            n_init=10,      # number of runs of the k-means algorithm with different initial centroid values
            max_iter=300,   # maximum number of iterations inside the k-means algorithm
            tol=1e-03,      # relative tolerance for judging convergence
            random_state=0) # the state of the random number generator used to initialize the centroid
y_km = km.fit_predict(X)    # compute cluster centers and predict indices for each data point


# plot the clusters that k-means identified 
# from the dataset and the centroids of the clusters
plt.scatter(X[y_km == 0, 0],   # x value of the graph
            X[y_km == 0, 1],   # y value of the graph
            s=50,              # size of the plot
            c='lightgreen',    # plot color
            marker='s',        # marker shape
            edgecolor='black', # plot line color
            label='Cluster 1') # label name
plt.scatter(X[y_km == 1, 0],
            X[y_km == 1, 1],
            s=50, c='orange',
            marker='o', edgecolor='black',
            label='Cluster 2')
plt.scatter(X[y_km == 2, 0],
            X[y_km == 2, 1],
            s=50, c='lightblue',
            marker='v', edgecolor='black',
            label='Cluster 3')
plt.scatter(km.cluster_centers_[:, 0],
            km.cluster_centers_[:, 1],
            s=250, marker='*',
            c='red', edgecolor='black',
            label='Centroids')
plt.legend(scatterpoints=1)
plt.grid()
plt.tight_layout()

# plt.savefig('images/k-means_clustering.png', dpi=300)
plt.show()


## Using the elbow method to find the optimal number of clusters 

print('Distortion: %.2f' % km.inertia_)

# plot distortion by number of clusters using elbow method
distortions = []
for i in range(1, 11):
    km = KMeans(n_clusters=i, 
                init='k-means++', 
                n_init=10, 
                max_iter=300, 
                random_state=0)
    km.fit(X)
    distortions.append(km.inertia_)
plt.plot(range(1, 11), distortions, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
plt.tight_layout()

# plt.savefig('images/dist_by_num_of_culsters.png', dpi=300)
plt.show()


## Quantifying the quality of clustering  via silhouette plots
import numpy as np
from matplotlib import cm
from sklearn.metrics import silhouette_samples

km = KMeans(n_clusters=3, 
            init='k-means++', 
            n_init=10, 
            max_iter=300,
            tol=1e-04,
            random_state=0)
y_km = km.fit_predict(X)

cluster_labels = np.unique(y_km)     # eliminate diplicates in elements of y_km
n_clusters = cluster_labels.shape[0] # return the length of the array

# calculate silhouette coefficient
silhouette_vals = silhouette_samples(X, y_km, metric='euclidean')
y_ax_lower, y_ax_upper = 0, 0
yticks = []
for i, c in enumerate(cluster_labels):
    c_silhouette_vals = silhouette_vals[y_km == c]
    c_silhouette_vals.sort()
    y_ax_upper += len(c_silhouette_vals)
    color = cm.jet(float(i) / n_clusters)   # set color value
    plt.barh(range(y_ax_lower, y_ax_upper), # draw a horizontal bar chart
             c_silhouette_vals,             # width of bar
             height=1.0,                    # bar height
             edgecolor='none',              # bar end color
             color=color)                   # bar color

    yticks.append((y_ax_lower + y_ax_upper) / 2.) # add display position of cluster label
    y_ax_lower += len(c_silhouette_vals)          # add bar width to base value
    
silhouette_avg = np.mean(silhouette_vals)                # mean value of silhouette coefficient
plt.axvline(silhouette_avg, color="red", linestyle="--") # draw a dashed line on the mean of the coefficients
plt.yticks(yticks, cluster_labels + 1)                   # show cluster labels
plt.ylabel('Cluster')
plt.xlabel('Silhouette coefficient')
plt.tight_layout()

# plt.savefig('images/silhouette_analysis.png', dpi=300)
plt.show()


### Comparison to "bad" clustering

km = KMeans(n_clusters=2,
            init='k-means++',
            n_init=10,
            max_iter=300,
            tol=1e-04,
            random_state=0)
y_km = km.fit_predict(X)

plt.scatter(X[y_km == 0, 0],
            X[y_km == 0, 1],
            s=50,
            c='lightgreen',
            edgecolor='black',
            marker='s',
            label='Cluster 1')
plt.scatter(X[y_km == 1, 0],
            X[y_km == 1, 1],
            s=50,
            c='orange',
            edgecolor='black',
            marker='o',
            label='Cluster 2')

plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
            s=250, marker='*', c='red', label='Centroids')
plt.legend()
plt.grid()
plt.tight_layout()

# plt.savefig('images/k-means_clustering_in_bad_condition.png', dpi=300)
plt.show()


# silhouette analyis in bad clustering condition
cluster_labels = np.unique(y_km)
n_clusters = cluster_labels.shape[0]
silhouette_vals = silhouette_samples(X, y_km, metric='euclidean')
y_ax_lower, y_ax_upper = 0, 0
yticks = []
for i, c in enumerate(cluster_labels):
    c_silhouette_vals = silhouette_vals[y_km == c]
    c_silhouette_vals.sort()
    y_ax_upper += len(c_silhouette_vals)
    color = cm.jet(float(i) / n_clusters)
    plt.barh(range(y_ax_lower, y_ax_upper), c_silhouette_vals, height=1.0, 
             edgecolor='none', color=color)

    yticks.append((y_ax_lower + y_ax_upper) / 2.)
    y_ax_lower += len(c_silhouette_vals)
    
silhouette_avg = np.mean(silhouette_vals)
plt.axvline(silhouette_avg, color="red", linestyle="--") 

plt.yticks(yticks, cluster_labels + 1)
plt.ylabel('Cluster')
plt.xlabel('Silhouette coefficient')

plt.tight_layout()

# plt.savefig('images/silhouette_analysis_in_bad_clustering.png', dpi=300)
plt.show()

# --------------------------------------------------------------------------------

#ml-source_29-30_s2.py

# Organizing clusters as a hierarchical tree
## Grouping clusters in bottom-up fashion

# make randam dataset
import pandas as pd
import numpy as np

np.random.seed(123)

variables = ['X', 'Y', 'Z']
labels = ['ID_0', 'ID_1', 'ID_2', 'ID_3', 'ID_4']

X = np.random.random_sample([5, 3])*10 # generate sample data with 5 rows and 3 columns
df = pd.DataFrame(X, columns=variables, index=labels)
print(df)


## Performing hierarchical clustering on a distance matrix

from scipy.spatial.distance import pdist, squareform

row_dist = pd.DataFrame(squareform(pdist(df, metric='euclidean')),
                        columns=labels,
                        index=labels)
print(row_dist)


# application of agglomerative hierarchical clustering
# based on complete linkage method
from scipy.cluster.hierarchy import linkage

#row_clusters = linkage(row_dist, method='complete', metric='euclidean') # DO NOT USE
#row_clusters = linkage(pdist(df, metric='euclidean'), method='complete') # OK
row_clusters = linkage(df.values, method='complete', metric='euclidean') # OK
print(pd.DataFrame(row_clusters,
             columns=['row label 1', 'row label 2',
                      'distance', 'no. of items in clust.'],
             index=['cluster %d' % (i + 1)
                    for i in range(row_clusters.shape[0])]))


# plot a dendrogram
# (uncomment the corresponding lines if you want to plot in black)
from scipy.cluster.hierarchy import dendrogram

# make dendrogram black (part 1/2)
# from scipy.cluster.hierarchy import set_link_color_palette
# set_link_color_palette(['black'])

row_dendr = dendrogram(row_clusters, 
                       labels=labels,
                       # make dendrogram black (part 2/2)
                       # color_threshold=np.inf
                       )
plt.tight_layout()
plt.ylabel('Euclidean distance')

# plt.savefig('images/dendrogram.png', dpi=300, 
#             bbox_inches='tight')
plt.show()


## Attaching dendrograms to a heat map

# plot row dendrogram
fig = plt.figure(figsize=(8, 8), facecolor='white')
axd = fig.add_axes([0.09, 0.1, 0.2, 0.6])

# note: for matplotlib < v1.5.1, please use orientation='right'
row_dendr = dendrogram(row_clusters, orientation='left')

# reorder data with respect to clustering
df_rowclust = df.iloc[row_dendr['leaves'][::-1]]

# generate a heatmap and place it to the right of the dendrogram
axm = fig.add_axes([0.23, 0.1, 0.6, 0.6])  # x-pos, y-pos, width, height
cax = axm.matshow(df_rowclust, interpolation='nearest', cmap='hot_r')

# remove axes spines from dendrogram
axd.set_xticks([])
axd.set_yticks([])
for i in axd.spines.values():
    i.set_visible(False)

# plot heatmap
fig.colorbar(cax)
axm.set_xticklabels([''] + list(df_rowclust.columns))
axm.set_yticklabels([''] + list(df_rowclust.index))

# plt.savefig('images/dendrogram_with_heatmap.png', dpi=300)
plt.show()


## Applying agglomerative clustering via scikit-learn

from sklearn.cluster import AgglomerativeClustering

ac = AgglomerativeClustering(n_clusters=3,         # num.of clusters
                             affinity='euclidean', # similarity index
                             linkage='complete')   # linkage method
labels = ac.fit_predict(X)
print('Cluster labels: %s' % labels)


# retry with 2 clusters
ac = AgglomerativeClustering(n_clusters=2, 
                             affinity='euclidean', 
                             linkage='complete')
labels = ac.fit_predict(X)
print('Cluster labels: %s' % labels)

exit()
