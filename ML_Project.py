from math import ceil

import pandas as pd
import pandas as pd
import numpy as np
import statsmodels as sm
import sklearn as skl
import sklearn.preprocessing as preprocessing
import sklearn.linear_model as linear_model
import sklearn.metrics as metrics
import sklearn.tree as tree
import seaborn as sns
import matplotlib.pyplot as plt
import scipy as sp

from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

data = pd.read_csv('data/adult.data.txt',
                   names=['Age', 'Workclass', 'fnlwgt', 'Education', 'Education-Num', 'Martial Status', 'Occupation',
                          'Relationship', 'Race', 'Sex', "Capital Gain", "Capital Loss", 'Hours per week', 'Country',
                          'Target'],
                   sep=r'\s*,\s*',
                   engine='python',
                   na_values="?")
original_data = data.copy()

fig = plt.figure(figsize=(20, 15))
cols = 5
rows = ceil(float(data.shape[1]) / cols)
for i, column in enumerate(data.columns):
    ax = fig.add_subplot(rows, cols, i + 1)
    ax.set_title(column)
    if data.dtypes[column] == np.object:
        data[column].value_counts().plot(kind="bar", axes=ax)
    else:
        data[column].hist(axes=ax)
        plt.xticks(rotation="vertical")
plt.subplots_adjust(hspace=0.7, wspace=0.2)

plt.show()

data = data.dropna()
print(data.isna().sum())
for i, column in enumerate(data[['Workclass', 'Education', 'Martial Status', 'Occupation', 'Relationship', 'Race',
                                 'Sex', 'Country', 'Target']].columns):
    l = list(set(data[column]))
    data[column] = data[column].map(lambda x: l.index(x))

# hmap =  data.corr().apply(lambda x: abs(x))
hmap = data.corr()

plt.subplots(figsize=(12, 9))
sns.heatmap(hmap, vmax=.8, annot=True, square=True)

plt.show()

X = data[['Education-Num', 'Age', 'Hours per week']].values
# X = data.drop(['Target', 'Relationship', 'Education'], axis=1)
y = data[['Target']].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=21, stratify=y)

clf = tree.DecisionTreeClassifier()
clf.fit(X_train, y_train)
predn = clf.predict(X_test)
print('The accuracy of the model is', metrics.accuracy_score(predn, y_test))

#svc = svm.SVC(kernel='linear', verbose= True)

#svc.fit(X_train, y_train)

#y_pred = svc.predict(X_test)

#print("Test set predictions:\n {}".format(y_pred))
#print(svc.score(X_test, y_test))


#Tuning the model



param_grid= {'n_neighbors': np.arange(1,80)}
knn = KNeighborsClassifier()
knn_cv=GridSearchCV(knn, param_grid, cv=5, verbose= True)
y = y.reshape(30162)
knn_cv.fit(X, y)
print(knn_cv.best_params_)
print(knn_cv.best_score_)