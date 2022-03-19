from statistics import mean
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.cluster import KMeans

#loading the data
df = pd.read_csv('Countryclusters.csv',sep=';')
print(df.columns.tolist())
print(df["Lat"])
plt.scatter(df["Lon"],df["Lat"])
plt.xlim(-180,180)
plt.ylim(-90,90)
plt.show()

x = df.iloc[:,0:2] # 1t for rows and second for columns

kmeans = KMeans(3)
kmeans.fit(x)

identified_clusters = kmeans.fit_predict(x)

data_with_clusters = df.copy()
data_with_clusters['Clusters'] = identified_clusters
plt.scatter(data_with_clusters['Lon'],data_with_clusters['Lat'],c=data_with_clusters['Clusters'],cmap='rainbow')
plt.show()