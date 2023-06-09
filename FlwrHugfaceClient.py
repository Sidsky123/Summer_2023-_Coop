#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf



# In[2]:


import torch
torch.cuda.is_available()


# In[3]:


help(torch.cuda)


# In[4]:


tf.test.is_built_with_cuda()


# In[5]:


import sys
import pandas as pd
import numpy as np
import sklearn
import matplotlib
import keras
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import seaborn as sns

cleveland = pd.read_csv('C:/Users/siddh/heart_statlog_cleveland_hungary_final.csv')


# In[6]:


print( 'Shape of DataFrame: {}'.format(cleveland.shape))
print (cleveland.loc[1])


# In[7]:


cleveland.head()


# In[8]:


data = cleveland[~cleveland.isin(['?'])]
data.loc[280:]
data = data.dropna(axis=0)


# In[9]:


print(data.shape)
print(data.dtypes)


# In[10]:


data = data.apply(pd.to_numeric)
data.dtypes


# In[11]:


data.describe()


# In[12]:


data.hist(figsize = (12, 12))
plt.show()


# In[13]:


X = np.array(data.drop(['target'],axis=1))
y = np.array(data['target'])


# In[14]:


X


# In[15]:


y


# In[16]:


mean = X.mean(axis=0)
X -= mean
std = X.std(axis=0)
X /= std


# In[17]:


X[0]


# In[ ]:


X.shape


# In[18]:


from sklearn import model_selection

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, stratify=y, random_state=42, test_size = 0.2)


# In[19]:


from keras.utils.np_utils import to_categorical

Y_train = to_categorical(y_train, num_classes=None)
Y_test = to_categorical(y_test, num_classes=None)
print (Y_train.shape)
print (Y_train[:10])


# In[ ]:


##Below is code for learning model through keras


# In[ ]:


# from keras.models import Sequential
# from keras.layers import Dense
# from keras.optimizers import Adam

# # define a function to build the keras model
# def create_model():
#     # create model
#     model = Sequential()
#     model.add(Dense(8, input_dim=11, kernel_initializer='normal', activation='relu'))
#     model.add(Dense(4, kernel_initializer='normal', activation='relu'))
#     model.add(Dense(2, activation='softmax'))
    
#     # compile model
#     adam = Adam(lr=0.001)
#     model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
#     return model

# model = create_model()

# print(model.summary())

# model.fit(X_train, Y_train, epochs=100, batch_size=10, verbose = 1)


# In[20]:


import utils


# In[21]:


X_train.shape, X_test.shape, y_train.shape, y_test.shape


# In[22]:


from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)


# In[23]:


from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(y_test, predictions))


# In[24]:


from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
## Accuracy
print('Accuracy = ', accuracy_score(y_test,predictions))
## Precision
print('Precision = ', precision_score(y_test,predictions))
## Recall
print('Recall = ', recall_score(y_test,predictions))
## F1 Score
print('F1_Score = ', f1_score(y_test,predictions))


# In[25]:


utils.set_initial_params(model)


# In[26]:


import warnings
import flwr as fl


# In[37]:


class ClientLogR(fl.client.NumPyClient):
    def get_parameters(self, config):  # type: ignore
        
        return utils.get_model_parameters(model)

    def fit(self, parameters, config):  # type: ignore
        utils.set_model_params(model, parameters)
        with warnings.catch_warnings():
            
            model.fit(X_train, y_train)
        print(f"Training finished for round {config['server_round']}")
       
        return utils.get_model_parameters(model), len(X_train), {}

    def evaluate(self, parameters, config):  # type: ignore
        utils.set_model_params(model, parameters)
        loss = log_loss(y_test, model.predict_proba(X_test))
        accuracy = model.score(X_test, y_test)
        
        return loss, len(X_test), {"accuracy": accuracy}


# In[38]:


fl.client.start_numpy_client(server_address="localhost:5006", client= ClientLogR())


 # In[ ]:




