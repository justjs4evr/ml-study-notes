###practiced Cancer data from sklearn

import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

#load the breast cancer dataset
cancer = datasets.load_breast_cancer()
## check cancer.head(), cancer.shape(), cancer.data.type()

#preprocessing: train test split
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, test_size=0.3, random_state=42,)

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

# train a knn classifier
knn = KNeighborsClassfier(n_neighbors=5, p=2, metric='minkowski')
knn.fit(X_train_std, y_train)
knn.score(X_test_std, y_test) ## 0.9590643274853801

#train a random forest classifier
rf = RandomForestClassifier( n_estimators=25, random_state=4, n_jobs=2, max_depth=10)
rf.fit(X_train, y_train)
rf.score(X_test, y_test) ## 0.9707602339181286

#train another random forest classifier with different hyperparameters
rf2 = RandomForestClassifier( n_estimators=25, max_depth=15, random_state=4, n_jobs=2)
rf2.fit(X_train, y_train)
rf2.score(X_test, y_test) ## 0.9649122807017544