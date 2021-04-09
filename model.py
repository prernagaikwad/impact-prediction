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


# In[3]:


#The method .copy() is used here so that any changes made in new DataFrame don't get reflected in the original one
inc=data.copy()


# In[4]:


#Replace ? with the nan
inc=inc.replace("?",np.nan)


# In[5]:


#Removing extra string
inc["ID_caller"]= inc["ID_caller"].str.replace("Caller", " ") 
inc["opened_by"]= inc["opened_by"].str.replace("Opened by", " ") 
inc["Created_by"]= inc["Created_by"].str.replace("Created by", " ") 
inc["Category Id"]= inc["Category Id"].str.replace("Subcategory", " ") 
inc["user_symptom"]=inc["user_symptom"].str.replace("Symptom", " ") 
inc["Support_group"]=inc["Support_group"].str.replace("Group", " ") 
inc["support_incharge"]=inc["support_incharge"].str.replace("Resolver", " ") 
inc["problem_ID"]=inc["problem_ID"].str.replace("Problem ID", " ") 
inc["updated_by"]= inc["updated_by"].str.replace("Updated by", " ") 


# In[6]:


#Rename the column name
inc.rename({'Category Id':'Category_id'},axis=1, inplace=True)


# In[7]:


inc["target_impact"]=inc["impact"].apply(lambda x: int(x.split(' ')[0]))


# In[8]:


inc["location"]= inc["location"].str.replace("Location", " ") 


# In[18]:


#format the date & columns
inc["updated_day"]=pd.to_datetime(inc.updated_at).dt.day
inc["updated_month"]=pd.to_datetime(inc.updated_at).dt.month
inc["updated_year"]=pd.to_datetime(inc.updated_at).dt.year
inc["updated_hr"]=pd.to_datetime(inc.updated_at).dt.hour
inc["updated_minute"]=pd.to_datetime(inc.updated_at).dt.minute
inc["opened_at_day"]=pd.to_datetime(inc.opened_time).dt.day
inc["opened_at_month"]=pd.to_datetime(inc.opened_time).dt.month
inc["opened_at_year"]=pd.to_datetime(inc.opened_time).dt.year
inc["opened_at_hr"]=pd.to_datetime(inc.opened_time).dt.hour
inc["opened_at_minute"]=pd.to_datetime(inc.opened_time).dt.minute
inc["created_at_day"]=pd.to_datetime(inc.created_at).dt.day
inc["created_at_month"]=pd.to_datetime(inc.created_at).dt.month
inc["created_at_year"]=pd.to_datetime(inc.created_at).dt.year
inc["created_at_hr"]=pd.to_datetime(inc.created_at).dt.hour
inc["created_at_minute"]=pd.to_datetime(inc.created_at).dt.minute


# In[19]:


inc2=inc.copy()


# In[20]:


from sklearn.preprocessing import LabelEncoder
number=LabelEncoder()
inc2['ID_status']=number.fit_transform(inc2['ID_status'])
inc2['ID_status'].value_counts()


# In[21]:


inc2['active']=number.fit_transform(inc2['active'])
inc2['active'].value_counts()


# In[22]:


inc2['type_contact']=number.fit_transform(inc2['type_contact'])
inc2['type_contact'].value_counts()


# In[23]:


inc2['Doc_knowledge']=number.fit_transform(inc2['Doc_knowledge'])
inc2['Doc_knowledge'].value_counts()


# In[24]:


inc2['confirmation_check']=number.fit_transform(inc2['confirmation_check'])
inc2['confirmation_check'].value_counts()


# In[25]:


#Filling na values with the median values
for columns in ['user_symptom','created_at_day','created_at_month','created_at_year','created_at_hr','created_at_minute']:
    median=inc2[columns].median()
    inc2[columns]=inc2[columns].fillna(median)


# In[26]:


inc2.drop(['impact'],axis=1,inplace=True)


# In[27]:


inc2.drop(['notify'],axis=1,inplace=True)


# In[28]:


inc2['ID']=inc2['ID'].str.replace("INC", " ") 
inc2.head(2)


# In[29]:


inc2['Created_by'] = inc2['Created_by'].astype(float)
inc2['Created_by'].fillna((inc2['Created_by'].mean()), inplace=True)
inc2['Created_by'] = inc2['Created_by'].astype(int)


# In[30]:


inc2['ID']=inc2['ID'].astype(int)
inc2['location']=inc2['location'].astype(int)
inc2['ID']=inc2['ID'].astype(float).astype(int)
inc2['ID_caller']=inc2['ID_caller'].astype(int)
inc2['opened_by']=inc2['opened_by'].astype(int)
inc2['updated_by']=inc2['updated_by'].astype(int)
inc2['location']=inc2['location'].astype(int)
inc2['Category_id']=inc2['Category_id'].astype(int)
inc2['user_symptom']=inc2['user_symptom'].astype(int)
inc2['Support_group']=inc2['Support_group'].astype(int)


# In[32]:


incident1=inc2.copy()


# In[38]:


incident1.drop(['opened_time','created_at','updated_at','support_incharge','change_request','problem_ID'],axis=1,inplace=True)


# In[39]:


incident1.info()


# In[40]:


X=incident1.drop("target_impact",axis=1)
y=incident1["target_impact"]


# In[41]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X, y, test_size=0.3,random_state=10)


# In[42]:


from imblearn.over_sampling import SMOTE
smote = SMOTE()


# In[43]:


X_train_smote,y_train_smote=smote.fit_resample(X_train,y_train)


# In[44]:


from collections import Counter
print("Before SMOTE :",Counter(y_train))
print("After SMOTE :",Counter(y_train_smote))


# In[45]:


X_train_new= X_train_smote[['opened_by','location','ID_caller','Category_id','ID']]
X_test_new= X_test[['opened_by','location','ID_caller','Category_id','ID']]


# In[95]:


X_train_new


# In[47]:


from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
import sklearn.metrics as metrics


# In[48]:


from xgboost import XGBClassifier
xgb = XGBClassifier()
xgb.fit(X_train_new, y_train_smote)
y_pred = xgb.predict(X_test_new)
accuracy = accuracy_score(y_test, y_pred)


# In[49]:


print('Accuracy of Bagging Classifier: {:.3f}'.format(accuracy))


# In[50]:


print("Train Accuracy:",xgb.score(X_train_new, y_train_smote)*100)
print("Test Accuracy:",xgb.score(X_test_new, y_test)*100)


# In[87]:


from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(max_depth=27)
classifier.fit(X_train_new,y_train_smote)


# In[88]:


# Predicting the model
Y_predict_rf =classifier.predict(X_test_new)


# In[89]:


from sklearn.metrics import classification_report
print(accuracy_score(y_test,Y_predict_rf))
print(classification_report(y_test,Y_predict_rf))


# In[90]:


print("Train Accuracy:",classifier.score(X_train_new, y_train_smote)*100)
print("Test Accuracy:",classifier.score(X_test_new, y_test)*100)


# In[91]:


import pickle
pickle_out = open("classifier.pkl","wb")
pickle.dump(classifier, pickle_out)
pickle_out.close()


# In[92]:


fv=[24,98,2403,164,86]


# In[93]:


fv = np.array(fv).reshape((1,-1))


# In[94]:


classifier.predict(fv)

