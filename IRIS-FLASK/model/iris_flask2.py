#!/usr/bin/env python
# coding: utf-8

# In[12]:


import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import naive_bayes


# In[6]:


iris = datasets.load_iris()


# In[8]:


iris.keys()


# In[9]:


X = iris.data
Y = iris.target


# In[10]:


x_train, x_test, y_train, y_test = train_test_split(X,Y, random_state = 0)


# In[18]:


NB = naive_bayes.GaussianNB()


# In[19]:


NB.fit(x_train,y_train)


# In[21]:


pred = NB.predict(x_test)


# In[24]:


classification = classification_report(pred, y_test)
print(classification)


# In[25]:


import pickle
from keras.models import model_from_json

model_json = NB.to_json()

with open('iris_model.json','wb') as json_file:
	json_file.write(model_json)

model.save_weights('iris_mode.h5')


# In[27]:




# In[ ]:




