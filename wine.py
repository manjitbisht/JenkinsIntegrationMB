#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 12:52:54 2019

@author: mbisht
"""

from __future__ import print_function
import os
import subprocess
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import accuracy_score
from sklearn import tree

#balance_data = pd.read_csv(
#'https://archive.ics.uci.edu/ml/machine-learning-databases/balance-scale/balance-scale.data',
#                           sep= ',', header= None)

#credit_data = pd.read_csv(
#https://archive.ics.uci.edu/ml/machine-learning-databases/credit-screening//crx.data',
 #                          sep= ',', header= None)
wine_quality = pd.read_csv(
'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv',
                           sep= ';')
print("* wine_quality.head()", wine_quality.head(), sep="\n", end="\n\n")

bins = (2, 6.5, 8)
group_names = ['bad', 'good']
wine=wine_quality['quality'] = pd.cut(wine_quality['quality'], bins = bins, labels = group_names)

from sklearn.preprocessing import StandardScaler, LabelEncoder
label_quality = LabelEncoder()
wine_quality['quality'] = label_quality.fit_transform(wine_quality['quality'])

X = wine_quality.drop('quality', axis = 1)
y = wine_quality['quality']
#print("Dataset Lenght:: ", len(balance_data))
#print("Dataset Shape:: ", balance_data.shape)
#print("Dataset:: ")
#balance_data.head()

#X = balance_data.values[:, 1:5]
#Y = balance_data.values[:,0]

#X_wine = wine_quality.values[:, 0:10]
#y_wine = wine_quality.values[:11]
X_wine = list(wine_quality.columns[:11])
#y_wine = wine_quality.columns[11]


dt = DecisionTreeClassifier(min_samples_split=20, random_state=99)
dt = dt.fit(X, y)

#dt = DecisionTreeClassifier(min_samples_split=20, random_state=99)
#dt.fit(X_wine, y_wine)

def visualize_tree(tree, feature_names):
    """Create tree png using graphviz.

    Args
    ----
    tree -- scikit-learn DecsisionTree.
    feature_names -- list of feature names.
    """
    with open("dt.dot", 'w') as f:
        export_graphviz(tree, out_file=f,
                        feature_names=feature_names)

    command = ["dot", "-Tpng", "dt.dot", "-o", "dt.png"]
    try:
        subprocess.check_call(command)
    except:
        exit("Could not run dot, ie graphviz, to "
             "produce visualization")

visualize_tree(dt, X_wine)


#X_train, X_test, y_train, y_test = train_test_split( X_wine, y_wine, test_size = 0.3, random_state = 100)



#clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100,
#                               max_depth=3, min_samples_leaf=5)
#clf_gini.fit(X_train, y_train)

#DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=3,
#            max_features=None, max_leaf_nodes=None, min_samples_leaf=5,
#            min_samples_split=2, min_weight_fraction_leaf=0.0,
 #           presort=False, random_state=100, splitter='best')
