import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

# Modifiable constants
filename = 'earthquake_formatted_data_partial_utm.txt' # Data source file located in data folder
outfile = 'earthquake_output_data_partial.txt' # Output file name
dist_min = 20000  # Initial # Tolerance in ~radius of points during clustering
dist_max = 20001  # Final #
dist_step = 1  # Step #
samps_min = 10  # Initial # Minimum number of samples required to constitute a cluster
samps_max = 10  # Final # 
samps_step = 10  # Step # 
scale_time_min = 4  # Initial # "4" equates 1 week (604k seconds) ~= 160km for clustering purposes
scale_time_max = 4  # Final #
scale_time_step = 4  # Step #
######################
dist_max = dist_max + dist_step if dist_max > dist_min else dist_min + dist_step
samps_max = samps_max + samps_step if samps_max > samps_min else samps_min + samps_step
scale_time_max = scale_time_max + scale_time_step if scale_time_max > scale_time_min else scale_time_min + scale_time_step

data_raw = np.loadtxt('data/' + filename,skiprows=1)
data = np.copy(data_raw)
labels = []
core_samples_mask_master = []
for dist in range(dist_min, dist_max, dist_step):
    for samps in range(samps_min, samps_max, samps_step):
        for scale_time in range(scale_time_min, scale_time_max, scale_time_step):
            data[:,0] = data_raw[:,0]/scale_time  # Transform seconds to be in line with distance scale of x and y

            clustering = DBSCAN(eps=dist, min_samples=samps, metric='euclidean').fit(data)

            # Mask out all but core sample data 
            core_samples_mask = np.zeros_like(clustering.labels_, dtype=bool)
            core_samples_mask[clustering.core_sample_indices_]= True
            core_samples_mask_master.append(core_samples_mask)

            # Get stats about clustering data
            labels.append(clustering.labels_)

output = [(np.append(data_raw[k], [row[k] for row in labels])) for k, m in enumerate(data_raw)]
np.savetxt("output/" + outfile, output, fmt="%.7f", header="time x y cluster_labels")
# Currently uses inital conditions
n_clusters_ = len(set(labels[0])) - (1 if -1 in labels[0] else 0)
n_noise_ = list(labels[0]).count(-1)
print("Number of noise points: ",n_noise_)
print("Number of clusters: ", n_clusters_)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# Assign color to each cluster
unique_labels = set(labels[0])
colors = [plt.cm.get_cmap(name='prism')(each)
        for each in np.linspace(0,1,len(unique_labels))]
# Plot each cluster iteratively onto figure
fig = plt.figure()
ax = plt.axes(projection='3d')
for k, col in zip(unique_labels, colors):
    color = [col]
    if k == -1:
        color = [[0,0,0,1]]
    class_member_mask = (labels[0] == k)
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
