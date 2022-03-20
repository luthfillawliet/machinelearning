from operator import le
from optparse import Values
from pydoc import cli
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import cluster
from sklearn.cluster import KMeans
#impor library untuk operator matematika
import math
#Library untuk memanipulasi array
import array

df = pd.read_csv('Data Pembayaran Rek ULP PNK_clean.csv',sep=';')

#df.sample(5)
import warnings

#print(df.info())
#Get length of array
numrows = len(df.index)
#print(numrows)

#read data ATM dan PPOB
dp = pd.read_excel('Data INDO,ALFA mart.xlsx')
jumlah_centroid = len(dp.index)
print("Jumlah centroid : ",jumlah_centroid)
plt.scatter(dp["Long"],dp["Lat"])
plt.show()

numrowsTraining = 9 #nanti kalau sudah bisa, ganti ke numrows
jumlah_centroidTraining = 9 #kalau sudah bisa, ganti ke Jumlah_centroid

#Siapkan Array penampung nilai
s = (jumlah_centroidTraining,2)
clustersR = np.zeros(s,dtype=int)

#Looping sebanyak jumlah data pelanggan
for i in range(numrowsTraining):
    #siapkan variabel penampung nilai hasil Euclidian Value
    evs = np.zeros(jumlah_centroidTraining,dtype=np.float64)
    #looping sebanyak jumlah titik centroid
    for j in range(jumlah_centroidTraining):
        #Menghitung jarak euclidian antara sampel dan centroid
        ev = math.sqrt((df['Lat'][i]-dp['Lat'][j])**2 + (df['Long'][i]-dp['Long'][j])**2)
        print("Nilai Euclidian ",ev)
        evs[j] = ev
    nilaimin = min(evs)
    print("Nilai min : ",nilaimin)
    #untuk mengetahui index array dari nilai paling min
    #print(evs)
    index_min = min(range(len(evs)),key=evs.__getitem__)
    print(index_min)
    #Masukkan Nilainya ke cluster
    clustersR = np.insert(clustersR)

clustersR[0][1] = 4
print(clustersR)

coba = [1,2,3],[2,3,4]
print(coba)


    #Masukkan data ke cluster sesuai index yang paling minimal



