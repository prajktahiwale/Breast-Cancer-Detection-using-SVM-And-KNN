#!/usr/bin/env python
# coding: utf-8

# Attribute Information:
1.Sample code number: id number
2.Clump Thickness: 1 - 10
3.Uniformity of Cell Size: 1 - 10
4.Uniformity of Cell Shape: 1 - 10
5.Marginal Adhesion: 1 - 10
6.Single Epithelial Cell Size: 1 - 10
7.Bare Nuclei: 1 - 10
8.Bland Chromatin: 1 - 10
9.Normal Nucleoli: 1 - 10
10.Mitoses: 1 - 10
11.Class: (2 for benign, 4 for malignant)
Malignant==> Cancerous

Benign==> Not Cancerous (Healthy)


# Background
All of our bodies are composed of cells. The human body has about 100 trillion cells within it. And usually those cells behave in a certain way. However, occasionally, one of these 100 trillion cells, behave in a different way and keeps dividing and pushes the other cells around it out of the way. That cell stops observing the rules of the tissue within which it is located and begins to move out of its normal position and starts invading into the tissues around it and sometimes entering the bloodstream and becoming is called a metastasis.

In summary, as we grow older,throughout a lifetime, we go through this knid of situation where a particular kind of gene is mutated where the protein that it makes is abnormal and drives the cell to behave in a different way that we call cancer.

This is what Dr. WIlliam H. Wolberg was observing and put together this dataset.
# predict whether a cell is Malignant or Benign

# In[ ]:


#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


data=pd.read_csv("C:/Users/aksha/Desktop/assignment/breastCancer.csv")


# In[4]:


data.head()


# # Data pre-processing

# In[5]:


data['class'].value_counts()


# In[6]:


data.dtypes #checking the data types of each column


# In[7]:


data['bare_nucleoli'] #let's inspect the 'bare_nucleoli' column


# In[8]:


data[data['bare_nucleoli']=='?'] #checking the presence of '?' in the 'bare_nucleoli' column


# In[9]:


data[data['bare_nucleoli']=='?'].sum() # alternatively


# Alternatively
# Using the isdigit() function

# In[10]:


digits_in_bare_nucleoli= pd.DataFrame(data.bare_nucleoli.str.isdigit())
digits_in_bare_nucleoli


# In[11]:


#df[digits_in_hp['horsepower'] == False] 
data[digits_in_bare_nucleoli['bare_nucleoli']== False]


# Let us replace these missing values with NaN

# In[12]:


df= data.replace('?', np.nan)


# In[13]:


df.bare_nucleoli


# In[14]:


df.median()


# In[15]:


df.head()


# In[16]:


df = df.fillna(df.median())


# In[17]:


df.dtypes


# In[18]:


df.bare_nucleoli


# In[19]:


df.dtypes


# In[20]:


df['bare_nucleoli'] = df['bare_nucleoli'].astype('int64')


# In[21]:


df.dtypes


# # Exploratory Data Analysis

# In[22]:


#dropping the index of the dataset
df.drop('id',axis=1,inplace=True)


# In[23]:


df.head()


# In[24]:


data.head()


# In[25]:


df.describe().T


# # Bivariate Data Analysis

# In[26]:


sns.distplot(df['class'])


# # Multivariate Data Analysis

# In[27]:


df.hist(bins=20, figsize=(30,30), layout=(6,3));


# In[28]:


plt.figure(figsize= (15,10))
sns.boxplot(data=df,orient="h")


# In[29]:


df.corr()


# In[30]:


#Heatmap of the correlation between the indepent attributes

plt.figure(figsize=(35,15))
sns.heatmap(df.corr(), vmax=1, square=True,annot=True,cmap='viridis')
plt.title('Correlation between different attributes')
plt.show()


# In[31]:


#Pairplot of the correlation/distribution between various independent attributes
sns.pairplot(df, diag_kind="kde")


# # Model Building

# In[32]:


df.head()


# In[34]:


# Dividing our dataset into training and testing set

X = df.drop('class', axis=1)  #selecting all the attributes except the class attribute
y = df['class'] #selecting class attribute. 


# In[35]:


#Splitting our data into 70:30
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1)


# # KNeighborsClassifier

# In[37]:


from sklearn.neighbors import KNeighborsClassifier

KNN = KNeighborsClassifier(n_neighbors= 5 , weights = 'distance' )


# In[38]:


# Call Nearest Neighbour algorithm

KNN.fit(X_train, y_train)


# In[39]:


predicted_1 = KNN.predict(X_test)
predicted_1


# In[40]:


from scipy.stats import zscore

print('KNeighborsClassifier Agorithm is predicting at {0:.2g}%'.format(KNN.score(X_test, y_test)*100))


# # Support Vector Machine

# In[42]:


from sklearn.svm import SVC


svc= SVC(gamma=0.025, C=3)
svc.fit(X_train, y_train)


# In[43]:


predicted_2 = svc.predict(X_test)
predicted_2


# In[44]:


print('SupportVectorClassifier Agorithm is predicting at {0:.2g}%'.format(svc.score(X_test, y_test)*100))


# In[45]:


knnPredictions=pd.DataFrame(predicted_1)
svcPredictions=pd.DataFrame(predicted_2)


# In[46]:


df1=pd.concat([knnPredictions,svcPredictions],axis=1)


# In[47]:


df1.columns=[['knnPredictions','svcPredictions']]


# In[48]:


df1


# In[49]:


from sklearn.metrics import classification_report

print("classification_report for KNN")

print("..."*10)

print(classification_report(y_test, predicted_1))


# In[50]:


from sklearn.metrics import classification_report

print("classification_report for SVC")

print("..."*10)

print(classification_report(y_test, predicted_2))


# In[51]:


from sklearn import metrics

print("Confusion Matrix For KNeighborsClassifier")
cm=metrics.confusion_matrix(y_test, predicted_1, labels=[2, 4])

df_cm = pd.DataFrame(cm, index = [i for i in [2,4]],
                  columns = [i for i in ["Predict M","Predict B"]])
plt.figure(figsize = (7,5))
sns.heatmap(df_cm, annot=True)


# In[52]:


print("Confusion Matrix For SupportVectorMachine")
cm=metrics.confusion_matrix(y_test, predicted_2, labels=[2, 4])

df_cm = pd.DataFrame(cm, index = [i for i in [2,4]],
                  columns = [i for i in ["Predict M","Predict B"]])
plt.figure(figsize = (7,5))
sns.heatmap(df_cm, annot=True)


# In[ ]:




