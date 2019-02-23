#importing the libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from statistics import mean
from statistics import mean




########## Reading the dataset
df = pd.read_csv('IrisData.txt', header=None)

#
#
########## Preprocessing the Data
#
#

# splitting Input from the dataset
X = df.iloc[:,:-1]

# converting the Input Data into matrix datastructure 
X = np.asmatrix(X)

# spilting the classification from the dataset
Y = df.iloc[:,4]

#encoding categorical data and converting it to the classification matrix.
Y1=np.zeros((150,1))
mapping = {'Iris-setosa':1, 'Iris-virginica':2, 'Iris-versicolor':3}
Y1=Y.replace(mapping)
Y1 = np.asmatrix(X)

#
#
########## K-fold Cross validation()
#
#

summ=[]
for k in range (0,10):
    # splitting the data set into training set and test set 
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y1, test_size=0.2, random_state = k)
    

    # Training the model using Linear Regresion 
    XT = np.transpose( X_train )
    XTX = np.matmul( XT, X_train )
    XTXINV = np.linalg.inv( XTX )
    XTXINVTR = np.dot( XTXINV, XT )
    beta = np.dot( XTXINVTR, Y_train )
    
    # Using Training Model to perform classification.
    Y_bar = np.matmul( X_test, beta )

    # calcualting Accuracy of the model.
    Y_bar=Y_bar.astype('int')
    cmp = ( Y_bar != Y_test )
    sumc= np.sum(cmp)
    count = len(Y_train)
    value = float (sumc )/ count
    summ.append(value)
    print "Accuracy for ", k, "th fold", (mean(summ)*100)

