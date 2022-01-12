#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np


# In[4]:


FireForest= pd.read_csv("F:/Dataset/forestfires.csv")


# In[5]:


FireForest


# In[6]:


seed=7


# In[7]:


np.random.seed(seed)


# In[10]:


FireForest.info()


# In[15]:


from  sklearn import preprocessing


# In[20]:


le = preprocessing.LabelEncoder()


# In[23]:


FireForest["month"]= le.fit_transform(FireForest['month'])


# In[24]:


FireForest["day"]= le.fit_transform(FireForest['day'])


# In[25]:


FireForest["size_category"]= le.fit_transform(FireForest['size_category'])


# In[26]:


FireForest


# In[27]:


X=FireForest.iloc[:,:11]


# In[28]:


Y=FireForest.iloc[:,-1]


# In[29]:


X


# In[30]:


Y


# In[40]:


import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow import keras
from tensorflow.keras import layers


# In[41]:


model= Sequential()


# In[42]:


model.add(layers.Dense(50, input_dim=11,  activation='relu'))


# In[43]:


model.add(layers.Dense(11,  activation='relu'))


# In[44]:


model.add(layers.Dense(1, activation='sigmoid'))


# In[45]:


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])


# In[46]:


history = model.fit(X, Y, validation_split=0.33, epochs=100, batch_size=10)


# In[47]:


scores = model.evaluate(X, Y)


# In[48]:


print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


# In[49]:


import matplotlib.pyplot as plt


# In[50]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[53]:


plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[54]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[ ]:




