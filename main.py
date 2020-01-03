print(__doc__)
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

filename = 'earthquake_formatted_data_partial_utm.txt'
data = np.loadtxt('data/' + filename,skiprows=1)
# newdata = StandardScaler().fit_transform(data)
data[:,0] = data[:,0]/5
clustering = DBSCAN(eps=10000, min_samples=5, metric='euclidean').fit(data)
core_samples_mask = np.zeros_like(clustering.labels_, dtype=bool)
core_samples_mask[clustering.core_sample_indices_]= True
labels = clustering.labels_

n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)
print("Number of noise points: ",n_noise_)
print("Number of clusters: ", n_clusters_)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
unique_labels = set(labels)
colors = [plt.cm.get_cmap(name='plasma')(each)
        for each in np.linspace(0,1,len(unique_labels))]
fig = plt.figure()
ax = plt.axes(projection='3d')
for k, col in zip(unique_labels, colors):
    color = [col]
    if k == -1:
        color = [[0,0,0,1]]
    class_member_mask = (labels == k)
    xy = data[class_member_mask & core_samples_mask]
    ax.scatter(xy[:,1], xy[:,2], xy[:,0], marker='o', s=17, c=color, edgecolor='k')
    xy = data[class_member_mask & ~core_samples_mask]
    ax.scatter(xy[:, 1], xy[:, 2], xy[:,0], marker='o', c=color,
             edgecolor='k', s=6)
# ax.scatter(data[:,1], data[:,2], data[:,0], s=1, c=[i for i,j in enumerate(labels)], cmap=plt.cm.get_cmap(name='Spectral'))
plt.show()