import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

# Modifiable constants
filename = 'earthquake_formatted_data_partial_utm.txt' # Data source file located in data folder
outfile = 'earthquake_output_data.txt' # Output file name
dist = 10000 # Tolerance in ~distance of points during clustering
samps = 5 # Number of samples required to constitute a cluster

data = np.loadtxt('data/' + filename,skiprows=1)
data[:,0] = data[:,0]/5 # Transform seconds to be in line with distance scale of x and y

clustering = DBSCAN(eps=dist, min_samples=samps, metric='euclidean').fit(data)

# Mask out all but core sample data
core_samples_mask = np.zeros_like(clustering.labels_, dtype=bool)
core_samples_mask[clustering.core_sample_indices_]= True

# Get stats about clustering data
labels = clustering.labels_
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)
print("Number of noise points: ",n_noise_)
print("Number of clusters: ", n_clusters_)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# Assign color to each cluster
unique_labels = set(labels)
colors = [plt.cm.get_cmap(name='plasma')(each)
        for each in np.linspace(0,1,len(unique_labels))]
# Plot each cluster iteratively onto figure
fig = plt.figure()
ax = plt.axes(projection='3d')
for k, col in zip(unique_labels, colors):
    color = [col]
    if k == -1:
        color = [[0,0,0,1]]
    class_member_mask = (labels == k)
    # Core cluster points made larger
    xy = data[class_member_mask & core_samples_mask]
    ax.scatter(xy[:,1], xy[:,2], xy[:,0], marker='o', s=17, c=color, edgecolor='k')
    xy = data[class_member_mask & ~core_samples_mask]
    ax.scatter(xy[:, 1], xy[:,2], xy[:,0], marker='o', s=6, c=color, edgecolor='k')
plt.show()
output = [(np.append(data[k], labels[k])) for k,m in enumerate(labels)]
np.savetxt("output/"+outfile, output, fmt="%.7f", header="time x y cluster_labels")