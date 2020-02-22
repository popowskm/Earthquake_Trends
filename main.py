import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

# Modifiable constants
filename = 'earthquake_formatted_data_partial_utm.txt' # Data source file located in data folder
outfile = 'earthquake_output_data_partial.txt' # Output file name
dist = 20000  # Tolerance in ~radius of points during clustering
samps = 10  # Minimum number of samples required to constitute a cluster
scale_time = 4  # "4" equates 1 week (604k seconds) ~= 160km for clustering purposes

data_raw = np.loadtxt('data/' + filename,skiprows=1)
data = data_raw
data[:,0] = data[:,0]/scale_time  # Transform seconds to be in line with distance scale of x and y

clustering = DBSCAN(eps=dist, min_samples=samps, metric='euclidean').fit(data)

# Mask out all but core sample data
core_samples_mask = np.zeros_like(clustering.labels_, dtype=bool)
core_samples_mask[clustering.core_sample_indices_]= True

# Get stats about clustering data
labels = clustering.labels_
output = [(np.append(data_raw[k], labels[k])) for k, m in enumerate(labels)]
np.savetxt("output/" + outfile, output, fmt="%.7f", header="time x y cluster_labels")

n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)
print("Number of noise points: ",n_noise_)
print("Number of clusters: ", n_clusters_)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# Assign color to each cluster
unique_labels = set(labels)
colors = [plt.cm.get_cmap(name='prism')(each)
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
    ax.scatter(xy[:,1], xy[:,2], xy[:,0]*scale_time, marker='o', s=17, c=color, edgecolor=color)
    xy = data[class_member_mask & ~core_samples_mask]
#    ax.scatter(xy[:, 1], xy[:,2], xy[:,0]*scale_time, marker='o', s=6, c=color, edgecolor=color)

ax.set_xlabel('UTM X')
ax.set_ylabel('UTM Y')
ax.set_zlabel('time')
ax.view_init(50,0)  # South to North section
# ax.view_init(90,-90)  # Top view
plt.show()
