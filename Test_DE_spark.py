#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pyspark.ml.clustering import KMeans 
from pyspark import SparkContext 
from pyspark.sql import SQLContext 
from pyspark.sql.functions import isnan, when, count, col 
from pyspark.ml.feature import StringIndexer 
import matplotlib.pyplot as plt 
from pyspark.ml.linalg import Vectors 
from pyspark.ml.feature import VectorAssembler 


# In[2]:


#get Spark connexion:
spc = SparkContext('local', 'Spark SQL') 
sqlc = SQLContext(spc)


# In[3]:


#Get the data
path = "Brisbane_CityBike.json"
sdf = sqlc.read.json(path)


# In[4]:


#Checking the state of our data and analysing 
print(sdf.describe().show())
sdf.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in sdf.columns]).show()


# In[19]:


# Applying Kmeans 

coordination =['latitude','longitude']
df = sdf.select("latitude", "longitude")
# regroup the coordination in one vector 
assembler = VectorAssembler(inputCols= coordination,outputCol="features")
X = assembler.transform(df)

#Initsializing the the number of clusters into 3
kmeans = KMeans().setK(3).setSeed(1) 


#train the model
model =  kmeans.fit(X)

#results of KMeans.
results = model.transform(X)

#print the results(10 first rows)
print(results.head(10))


# In[ ]:




