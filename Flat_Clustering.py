
#k_mean

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from matplotlib import style
style.use("ggplot")


my_input= np.array([[4,2], [6,6], [2.6,4], [7,8], [3.5,5], [6,11]])

my_model= KMeans(n_clusters=3)                   #how many cluster I want to make
my_model.fit(my_input)

print("cluster_centers :\n", my_model.cluster_centers_)
print("labels : ", my_model.labels_)

colors = ['g.', 'r.', 'c.', 'y.']

plt.scatter=(my_input[:,0], c= my_model.labels_)
plt.scatter(my_model.cluster_centers_[:,1], marker='x', s=250, linewidths=5)

plt.show
