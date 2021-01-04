#!/usr/bin/env python
# coding: utf-8

# ### End-to-End Machine Learning Project

# In[4]:


#import the libraries 
import matplotlib.pyplot as plt #For Visualization of data
import pandas as pd #Data manuplation and Analysis
import numpy as np # for multi-dimensional array and  matrcies
df=pd.read_csv(r"https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv")
#the California Housing Prices dataset from the StatLib repository


# In[5]:


#show the data 
df.head()


# In[7]:


#checking how many catageriocal values present in the ocean_proximity feature
df['ocean_proximity'].value_counts()


# In[8]:


df.describe()


# In[9]:


#Visualize the feature of the data
get_ipython().run_line_magic('matplotlib', 'inline')
df.hist(bins=50,figsize=(20,15))


# In[13]:


#split the data into training and test data
from sklearn.model_selection import train_test_split
X=df.iloc[:,:-1].values
y=df.iloc[:,-1:].values
X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=0.8)


# In[21]:


# median_income feature converting from numerical to catagerocial feature
df['income_cat']=pd.cut(df['median_income'],bins=[0.0,1.5,3.0,4.5,6.0,np.inf],labels=[1,2,3,4,5])
df['income_cat'].hist()
plt.xlabel("Categories")
plt.ylabel("No of people")


# In[24]:


#Stratified Sampling based  on the income Category
from sklearn.model_selection import StratifiedShuffleSplit
split=StratifiedShuffleSplit(n_splits=1,test_size=0.2, random_state=42)
for train_index,test_index in split.split(df,df["income_cat"]):
    s_train_set=df.loc[train_index]
    s_test_set=df.loc[test_index]
    
s_test_set['income_cat'].value_counts()/len(s_test_set)
    


# In[25]:



df_c=s_test_set.copy()
df_c.plot(kind="scatter",x="longitude",y="latitude")


# In[26]:


df_c.plot(kind="scatter",x="longitude",y="latitude",alpha=0.1)


# In[28]:


df_c.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
 s=df_c["population"]/100, label="population", figsize=(10,7),
 c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
)
plt.legend()


# In[30]:


corr_matrix = df_c.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)

