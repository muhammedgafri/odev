#!/usr/bin/env python
# coding: utf-8

# Ad soyad: MuhammedGafri
# Okul NO: 2010235508
# Tıp Müh. 30%

# In[1]:


import pandas as pd
import numpy as np
df=pd.read_csv("breast-cancer.csv")
print(df)
print(df.info())


# In[2]:


print(df.describe())


# In[3]:


print(df.isnull().sum())


# In[6]:


print(df[df.duplicated()])


# In[7]:


df.drop_duplicates(inplace=True, keep="first")


# In[8]:


print(df[df.duplicated()])


# In[9]:


print(df)


# In[10]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# In[11]:


df=df.drop(["diagnosis"], axis=1)


# In[12]:


print(df)


# In[13]:


x=df.drop('id', axis=1)  
y=df['id']


# In[14]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)


# In[15]:


mlr=LinearRegression()
dt=DecisionTreeRegressor(random_state=42)
rf=RandomForestRegressor(random_state=42)


# In[18]:


mlr.fit(x_train,y_train)
dt.fit(x_train,y_train)
rf.fit(x_train,y_train)


# In[20]:


mlr_pred=mlr.predict(x_test)
dt_pred=dt.predict(x_test)
rf_pred=rf.predict(x_test)


# In[21]:


mlr_r2=r2_score(y_test, mlr_pred)
mlr_mae=mean_absolute_error(y_test, mlr_pred)
mlr_mse=mean_squared_error(y_test, mlr_pred)

dt_r2=r2_score(y_test, dt_pred)
dt_mae=mean_absolute_error(y_test, dt_pred)
dt_mse=mean_squared_error(y_test, dt_pred)

rf_r2=r2_score(y_test, rf_pred)
rf_mae=mean_absolute_error(y_test, rf_pred)
rf_mse=mean_squared_error(y_test, rf_pred)


# In[22]:


print("mlr","r2", mlr_r2, "mae", mlr_mae, "mse", mlr_mse)
print("dt", "r2" , dt_r2, "mae", dt_mae, "mse", dt_mse)
print("rf", "r2", rf_r2, "mae", rf_mae, "mse", rf_mse)


# In[ ]:




