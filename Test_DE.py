#!/usr/bin/env python
# coding: utf-8

# In[14]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sn
from sklearn.cluster import KMeans
from scipy.cluster.vq import kmeans,vq
from scipy.spatial.distance import cdist


# In[15]:


# Reading the data into panda Dataframe
df = pd.read_json("Brisbane_CityBike.json")
df.head()


# In[16]:


# Checking the state of our data and analysing 
df.info()


# In[17]:


print(df.describe())


# In[19]:


# the data seems to be clean ( no missing values and no NULL values)
# The feature "Name" is already composed of the rest of features of our data (location and number) 
# it will be more intersting if we make the clustering by only basing on the location of the stations ( latitude, longtitude)


# In[20]:


#Extracting the coordinates of the location
coord = df.loc[:,['latitude','longitude']]


# In[21]:


#Find the right number of clusters by performing kmeans
K = range(1,11)  
#kmeans from 1 to 10
KM = [kmeans(coord,k) for k in K] 
centroids = [cent for (cent,var) in KM]   # cluster centroids
#calculate Euclidian distance between the stations to measure similarity
Dk = [cdist(coord, cent, 'euclidean') for cent in centroids]

dist = [np.min(D,axis=1) for D in Dk]
avgWithinSS = [sum(d)/coord.shape[0] for d in dist]  


# In[22]:



# plot curve to visualize the optimized number of cluster
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(K, avgWithinSS, 'b.-')
ax.plot(K[2], avgWithinSS[2], 'r.')
plt.xlabel('Number of clusters')
plt.ylabel('Average number in cluster sum of squares')
plt.title(str('Optimized number of Clusters = ') + str(K[2]))  
plt.show()


# In[27]:


#K-Means clustering
nb_cluster = 3   #number of clusters
kmeans = KMeans(n_clusters=nb_cluster, random_state=1).fit(coord)
label=kmeans.labels_


# In[29]:


#visualization of the clustering results
plt.ylabel('Longitude')
plt.xlabel('Latitude')
for i in range(nb_cluster):
    cluster=np.where(label==i)[0]
    plt.plot(coord.latitude[cluster].values,coord.longitude[cluster].values,'*')
plt.title('Clustering based on coordination of bike stations')  
plt.show()


# In[ ]:




