#!/usr/bin/env python
# coding: utf-8

# In[218]:


import pandas as pd
import numpy as np
import os

import matplotlib.pyplot as plt
from matplotlib import rcParams


# In[219]:


data = pd.read_csv(r"./Dataset/Dataset.csv")


# In[220]:


data = data.replace('?', np.nan)


# In[221]:


data.head()


# In[222]:


print("***Number of missing values***\n")
print(data.isna().sum())


# In[223]:


df=data.fillna(data.median())


# In[224]:


print("***After Removing missing values***\n")
print(df.isna().sum())


# In[225]:


Dataset=df.to_csv(os.path.join(r'./preprocess_dataset/Dataset.csv'), index=False)
Dataset = pd.read_csv(r'./preprocess_dataset/Dataset.csv')


# In[226]:


Dataset.head()


# In[227]:


Dataset.tail()


# In[228]:


Dataset.info()


# In[229]:


Dataset.describe()


# In[231]:


rcParams['figure.figsize'] = 8,6
plt.bar(df['COVID-19'].unique(), df['COVID-19'].value_counts(), color = ['#50C878', '#E45328'])
plt.xticks([0, 1])
plt.xlabel('Target Classes')
plt.ylabel('Count')
plt.title('Count of each Target Class')


# In[ ]:




