
# coding: utf-8

# In[109]:


#importing the libraries
from IPython import get_ipython
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
get_ipython().run_line_magic('matplotlib', 'inline')


# In[111]:


#importing the dataset
flowers = pd.read_csv('irisdata.csv', header=None)
print(flowers)


# In[112]:


#splitting and printing
X = flowers.iloc[:,:-1]
#X= X[:,1:]
#X = X.astype(np.float64)

Y = flowers.iloc[:,:4]
#Y = Y.astype(np.float64)
print(X)
print(Y)


# In[113]:


#data visualization
sns.heatmap(flowers.corr())


# In[122]:


#encoding categorical data
'''from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder=LabelEncoder()
Y[:,0]=labelencoder.fit_transform(Y[:,0])

onehotencoder=OneHotEncoder(categorical_features=[3])
Y=onehotencoder.fit_transform(Y).toarray()
print(Y)
'''
Y1=np.zeros((150,1))
mapping = {'Iris-setosa':1, 'Iris-virginica':3, 'Iris-versicolor':2}
Y1=Y.replace(mapping)
print(Y1)
#Y1 = Y1.astype(np.float64)


# In[150]:


#splitting the data set into training set and test set
#X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2 ,random_state=0)
from statistics import mean
summ=[]
for n in range (0,5):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y1, test_size=0.2 ,random_state=n)
    XT=np.transpose(X_train)
    XTX=np.matmul(XT,X_train)
    XTXINV=np.linalg.inv(XTX)
    XTXINVTR=np.matmul(XTXINV,XT)
    beta=np.matmul(XTXINVTR,Y_train)
    
    Y_bar=np.matmul(X_test,beta)
    print(Y_bar)
    Y_bar=Y_bar.astype('int')
    print(Y_bar)
    #print(pd.to_numeric(Y_bar, downcast='unsigned'))
    cmp=(Y_bar!=Y_test)
    print(cmp)
    summ.insert(n, (np.sum(cmp)/Y_test.count()))
print(mean(summ)*100)

