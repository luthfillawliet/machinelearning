import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

df = pd.read_csv('olahan.csv',sep=';')

#df.sample(5)
import warnings
warnings.filterwarnings('ignore')
plt.figure(figsize=(16,5))
plt.subplot(1,2,1)
sns.distplot(df['Lat'])
plt.subplot(1,2,2)
sns.distplot(df['Long'])
plt.show()

latHighestAllowed = -5.10
latLowestAllowed = -5.2
longHighestAlllowed = 119.5
longLowestAllowed = 119
print("Highest allowed Latitude",latHighestAllowed)
print("Lowest allowed Latitude",latLowestAllowed)

print("Highest allowed Longitude",longHighestAlllowed)
print("Lowest allowed Longitude",longLowestAllowed)

#Finding Outliers for Latitude
df[(df['Lat'] > latHighestAllowed) | (df['Lat'] < latLowestAllowed)]
#Trimming Outlier
new_df = df[(df['Lat'] < latHighestAllowed) & (df['Lat'] > latLowestAllowed)]

#Finding Outliers for Longitude
df[(df['Lat'] > longHighestAlllowed) | (df['Lat'] < longLowestAllowed)]
#Trimming Outlier
new_df = df[(df['Lat'] < latHighestAllowed) & (df['Lat'] > longLowestAllowed)]

#Capping on Outliers Latitude
Lat_upper_limit = latHighestAllowed
Lat_lower_limit = latLowestAllowed
#Capping on Outliers Longitude
Long_upper_limit = longHighestAlllowed
Long_lower_limit = longLowestAllowed

#Now, apply the Capping on Lat
df['Lat'] = np.where(
    df['Lat']>Lat_upper_limit,
    Lat_upper_limit,
    np.where(
        df['Lat']<Lat_lower_limit,
        Lat_lower_limit,
        df['Lat']
    )
)
#Now, apply the Capping on Long
df['Long'] = np.where(
    df['Long']>Long_upper_limit,
    Long_upper_limit,
    np.where(
        df['Long']<Long_lower_limit,
        Long_lower_limit,
        df['Long']
    )
)
print(df['Lat'].describe())
print(df['Long'].describe())

# warnings.filterwarnings('ignore')
# plt.figure(figsize=(16,5))
# plt.subplot(1,2,1)
# sns.distplot(df['Lat'])
# plt.subplot(1,2,1)
# sns.displot(df['Long'])
# plt.show()

plt.scatter(df["Long"],df["Lat"])
plt.show()

x = df.iloc[:,32:34] # 1t for rows and second for columns
kmeans = KMeans(3)
kmeans.fit(x)

identified_clusters = kmeans.fit_predict(x)

data_with_clusters = df.copy()
data_with_clusters['Clusters'] = identified_clusters
plt.scatter(data_with_clusters['Long'],data_with_clusters['Lat'],c=data_with_clusters['Clusters'],cmap='rainbow')
plt.show()
