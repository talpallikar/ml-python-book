from sklearn import datasets as ds
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron

#load the iris dataset
iris = ds.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target

#split the dataset to 30% test and 70% train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

#standardize features by getting z scores
sc = StandardScaler()
sc.fit(X_train) #get standard dev and mean
X_train_std = sc.transform(X_train) #use standard dev and mean to calculate z score
X_test_std = sc.transform(X_test)   #use sd and mean to get z score

#train a perceptron from scikit
model = Perceptron(n_iter=40, eta0=0.1, random_state=0)
model.fit(X_train_std, y_train)

y_pred = model.predict(X_test_std)
print((y_test != y_pred).sum())