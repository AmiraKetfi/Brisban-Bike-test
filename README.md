**Bike Station Clustering with Python and PySpark**
This Repo contains two scripts to perform clustering of Bike station located in Brisbane, using features that have being extracted from a provided dataset.
 - **Data** : 
 The Data is a Json file named "Brisbane_CityBike.json", it contains information about Bike stations in Birsbane. These informations are mainly : The name of the station, number of bikes per station and location ( adresse, latitude and longitude).
 
 - **Clustring Method**:
 The file "Test_DE.py" is a python script to analyse and perform clustering on data using Kmeans method with calculating similarity using the euclidean function. You can have better visualisation and details about the method and function on the notebook "Test_DE.ipynb".
The file "Test_DE_spark.py" is a python script that performs the same clustering method while using pySpark, you can also find the notebook matching the script "Test_DE_spark.ipynb".
 - **How to run the sript**
 To be able to run the script you should already have python on your machine then, you should install all the requirements by ruing the command bellow : 
 

    sudo pip install -r requirements.txt
Then you should be able to run the script with the command bellow : 

    sudo python Test_DE.py

