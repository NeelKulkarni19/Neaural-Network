#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np


# In[2]:


GasTurbine= pd.read_csv('F:/Dataset/gas_turbines.csv')


# In[3]:


GasTurbine


# In[9]:


x = GasTurbine.drop(columns = ['TEY'], axis = 1) 


# In[10]:


x


# In[11]:


y = GasTurbine.iloc[:,7]


# In[12]:


y


# In[14]:


from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split


# In[15]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.1, random_state = 42)


# In[16]:


x_train_scaled = scale(x_train)


# In[17]:


x_test_scaled = scale(x_test)


# In[18]:


x_test_scaled


# In[19]:


inputsize = len(X.columns)


# In[20]:


outputsize = 1


# In[21]:


hiddenlayersize = 50


# In[22]:


import tensorflow as tf


# In[27]:


model = tf.keras.Sequential([tf.keras.layers.Dense(hiddenlayersize, input_dim = inputsize, activation = 'relu'),tf.keras.layers.Dense(hiddenlayersize, activation = 'relu'),tf.keras.layers.Dense(hiddenlayersize, activation = 'relu'),tf.keras.layers.Dense(outputsize)])


# In[28]:


optimizer = tf.keras.optimizers.SGD(learning_rate = 0.03)


# In[29]:


model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['MeanSquaredError'])


# In[31]:


numepochs = 50


# In[32]:


earlystopping = tf.keras.callbacks.EarlyStopping(patience = 2)


# In[33]:


model.fit(x_train_scaled, y_train, callbacks = earlystopping, validation_split = 0.1, epochs = num_epochs, verbose = 2)


# In[35]:


test_loss, mean_squared_error = model.evaluate(x_test_scaled, y_test)


# In[36]:


predictions = model.predict_on_batch(x_test_scaled)


# In[37]:


import matplotlib.pyplot as plt


# In[38]:


plt.scatter(y_test, predictions)


# In[39]:


predictions_df = pd.DataFrame()


# In[40]:


predictions_df['Actual'] = y_test


# In[41]:


predictions_df['Predicted'] = predictions


# In[42]:


predictions_df['% Error'] = abs(predictions_df['Actual'] - predictions_df['Predicted'])/predictions_df['Actual']*100


# In[43]:


predictions_df.reset_index(drop = True)


# In[ ]:




