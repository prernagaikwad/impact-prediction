#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Import the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

#To remove warning
import warnings
warnings.filterwarnings('ignore')


# In[2]:


#Load the dataset
data=pd.read_excel("sample_service.xlsx")


# In[4]:


#The method .copy() is used here so that any changes made in new DataFrame don't get reflected in the original one
inc=data.copy()


# In[5]:


#To head the first 5 rows
inc.head()


# In[6]:


inc.describe()


# In[7]:


inc.info()


# In[ ]:


# To remove the extra string
inc["ID_caller"]= inc["ID_caller"].str.replace("Caller", " ") 
inc["opened_by"]= inc["opened_by"].str.replace("Opened by", " ") 
#inc["Created_by"]= inc["Created_by"].str.replace("Created by", " ") 
inc["Category Id"]= inc["Category Id"].str.replace("Subcategory", " ") 
inc["user_symptom"]=inc["user_symptom"].str.replace("Symptom", " ") 
inc["Support_group"]=inc["Support_group"].str.replace("Group", " ") 
inc["support_incharge"]=inc["support_incharge"].str.replace("Resolver", " ") 
inc["problem_ID"]=inc["problem_ID"].str.replace("Problem ID", " ") 


# In[52]:


#Rename the column
inc.rename({'Category Id':'Category_id'},axis=1, inplace=True)
inc.head(2)


# In[53]:


#To find the null values
inc.isnull().sum()


# In[54]:


import seaborn as sns
cols = inc.columns 
colours = ['#000099', '#ffff00'] # specify the colours - yellow is missing. blue is not missing.
sns.heatmap(inc[cols].isnull(),
            cmap=sns.color_palette(colours))


# In[55]:


#Replace ? with the nan values
inc=inc.replace("?",np.nan)
inc.head(2)


# In[18]:


import seaborn as sns
cols = inc.columns 
colours = ['#000099', '#ffff00'] # specify the colours - yellow is missing. blue is not missing.
sns.heatmap(inc[cols].isnull(),
            cmap=sns.color_palette(colours))


# In[56]:


inc['Created_by'] = inc['Created_by'].astype(float)
inc['Created_by'].fillna((inc['Created_by'].mean()), inplace=True)


# In[57]:


inc['Created_by'].dtypes


# In[58]:


inc['created_at'].isnull().sum()


# In[59]:


inc['created_at'] =inc['created_at'].fillna(0)
inc.info()


# In[60]:


import seaborn as sns
cols = inc.columns 
colours = ['#000099', '#ffff00'] # specify the colours - yellow is missing. blue is not missing.
sns.heatmap(inc[cols].isnull(),
            cmap=sns.color_palette(colours))


# ### Visulaisation 

# In[20]:


plt.figure(figsize=(15,8))

sns.countplot(data['ID_status'])


# In[21]:


sns.countplot(inc['active'])


# In[22]:


sns.countplot(inc['Doc_knowledge'])


# In[23]:


sns.countplot(inc['confirmation_check'])


# In[24]:


sns.countplot(inc['notify'])


# In[25]:


sns.countplot(inc['type_contact'])


# In[26]:


sns.countplot(inc['impact'])


# In[27]:


inc['Waitingtime'] = inc['updated_at'] -inc['opened_time']
inc['Waitingtime']


# In[32]:


sns.countplot(inc['Waitingtime'])


# In[34]:


def dist(inc,var1):
    plt.figure()
    sns.distplot(inc[var1],kde = False,bins = 30)
    plt.show()


# In[35]:


dist(inc,"count_reassign")


# In[36]:


dist(inc, "count_opening")
open_val_count = inc['count_opening'].value_counts()
print(open_val_count)
print(open_val_count[0]/len(inc))


# In[37]:


dist(inc, "count_updated")
updated_val_count = inc['count_updated'].value_counts()
print(updated_val_count[0:20])
print("Most updated count :", updated_val_count.index.max())
print(updated_val_count[0:10].sum()/len(inc))


# In[38]:


inc['active'].value_counts()


# In[39]:


inc['count_reassign'].value_counts()


# In[40]:


inc['count_opening'].value_counts()


# In[41]:


inc['count_updated'].value_counts()


# In[42]:


inc['Doc_knowledge'].value_counts()


# In[43]:


inc['impact'].value_counts()


# In[44]:


inc['confirmation_check'].value_counts()


# In[45]:


inc['notify'].value_counts()


# In[46]:


inc['type_contact'].value_counts()


# In[47]:


inc.corr()


# In[49]:


f, ax = plt.subplots(figsize=(15,15))
corr = inc.corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool),annot = True, cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax)


# In[66]:


inc["impact"]=inc["impact"].str.replace("Medium", " ") 
inc["impact"]=inc["impact"].str.replace("Low", " ") 
inc["impact"]=inc["impact"].str.replace("High", " ") 
inc["impact"]=inc["impact"].str.replace("-", " ") 


# In[61]:


#Make a copy of a file
incident=inc.copy()


# In[ ]:


#inc.drop(columns = ['problem_ID','change_request','support_incharge'], inplace = True)


# In[ ]:





# ### Transformation

# In[ ]:


#modified_inc = pd.get_dummies(data = inc, columns = ['type_contact', 'Doc_knowledge', 'active', 'confirmation_check', 'notify', 'ID_status'], drop_first = True)


# In[68]:


import pandas_profiling as pp


# In[71]:


profile = pp.ProfileReport(inc) 
profile.to_file("output1.html")


# In[ ]:




