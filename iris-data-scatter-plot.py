#!/usr/bin/env python
# coding: utf-8

# In[34]:


import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("python/iris.data.csv") #please edit and direct ur data file here

groups = data.groupby('Class')

#predict data
pData = [5.7,2.6,5.0,2.0] #please edit your unknown data

# Plot
fig, ax = plt.subplots()
ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling

for name, group in groups:
    ax.plot(group.SW, group.PW, marker='o', linestyle='', ms=12, label=name)

    
ax.plot(pData[1], pData[3], marker='x', linestyle='', ms=12) #plot experimental data

ax.legend()
ax.set_xlabel('SW')
ax.set_ylabel('PW')
plt.show

# prepare data
# Import LabelEncoder
from sklearn import preprocessing

#creating labelEncoder
le = preprocessing.LabelEncoder()

SL = data.SL
SW = data.SW
PL = data.PL
PW = data.PW
flowerClass = data.Class

label=le.fit_transform(flowerClass)

#combinig weather and temp into single listof tuples
features=list(zip(SL,SW,PL,PW))

#kNN classifier
from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors=5)

# Train the model using the training sets
model.fit(features,label)

#Predict Output
predicted= model.predict([pData])
print('unknown data: ' + str(pData))
print('Classify unknow data as: ' + le.inverse_transform(predicted)[0]) #print(le.inverse_transform(predicted))


# In[ ]:




