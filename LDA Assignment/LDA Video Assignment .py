#!/usr/bin/env python
# coding: utf-8

# ## Linear Discriminant Analysis (LDA) Assignment
# 

# ## 1. The given dataset "Parkinson_Dataset_2" contains data from class 0 and 1. Split the dataset into 90:10 train - test ratio.

# In[60]:


import numpy as np
import pandas as pd 
import sklearn.model_selection as spltfunc
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv('Parkinson_Dataset_2.csv')
x=df.iloc[::,2:755]
y=df['class']
X_train, x_test, Y_train, y_test = spltfunc.train_test_split(x, y, train_size=0.90,test_size=0.10, random_state=2)


# In[61]:


x


# In[62]:


y


# In[63]:


X_train


# In[64]:


plt.figure(figsize=(12,10))
cor = df.corr()
sns.heatmap(cor)
plt.show()


# In[65]:


trydf=df.iloc[::,0:10]
#trydf=df
sns.heatmap(trydf.corr())


# ## 2. PerformLDAonthedatasetonthe90%split.

# In[66]:


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.tree import DecisionTreeClassifier # Used for checking the performance of LDA


# In[70]:


tree = DecisionTreeClassifier(criterion="entropy")
tree.fit(X_train,Y_train)
accuracy = tree.score(x_test,y_test)
print("Accuracy",accuracy*100)


# In[67]:


lda=LDA(n_components=1)
lda.fit(X_train, Y_train)


# In[68]:


lda.score(x_test,y_test)


# ## 3. Test the accuracy on the 10% split

# In[71]:


# Fit the method's model
lda.fit(X_train, Y_train)

# Fit a Decision Tree classifier on the embedded training set
tree.fit(lda.transform(X_train), Y_train)

# Compute the Decision Tree accuracy on the embedded test set
acc = tree.score(lda.transform(x_test), y_test)



# In[72]:


acc*100


# In[ ]:




