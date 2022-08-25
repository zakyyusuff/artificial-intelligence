# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 23:24:43 2022

@author: Danang
"""

import pandas as pd
import numpy as np
np.random.seed(1)
data6 = pd.DataFrame({"C" : np.random.randint(low=1, high=100, size=500),
                      "D" : np.random.normal(0.0, 0.1, size=500)
                      })
#%%
print(data6)

#%%
data6.to_csv('DanangAriSubarkah.csv')

#%%
d_train=data6[:450]
d_test=data6[450:]

#%% 3
import pandas as pd
d = pd.read_csv("Eminem.csv")
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
dvec = vectorizer.fit_transform(d['CONTENT'])
dvec
daptarkata=vectorizer.get_feature_names()
dshuf = d.sample (frac =1)
d_train3=dshuf [:300]
d_test3=dshuf [300:]
d_train_att=vectorizer.fit_transform(d_train3 ['CONTENT'])
d_train_att
d_test_att=vectorizer.transform(d_test3 ['CONTENT'])
d_test_att
d_train_label=d_train3['CLASS']
d_test_label=d_test3['CLASS']

#%% 4
from sklearn import svm
clfsvm = svm.SVR(gamma = 'auto')
clfsvm.fit(d_train_att, d_train_label)
print(clfsvm.fit(d_train_att, d_train_label))

#%% 5
from sklearn import tree
clftree = tree.DecisionTreeClassifier()
clftree.fit(d_train_att, d_train_label)
print(clftree.fit(d_train_att, d_train_label))

#%% 6
from sklearn.metrics import confusion_matrix
pred_labels= clftree.predict(d_test_att)
cm=confusion_matrix(d_test_label, pred_labels)
print(pred_labels)
print(cm)

#%%
import matplotlib.pyplot as plt
import itertools
def plot_confusion_matrix (cm, classes, normalize=False, 
                           title='Confusion_matrix', 
                           cmap=plt.cm.Blues) :
    if normalize :
        cm = cm.astype ('float') / cm.sum(axis =1)[:, np.newaxis]
        print( "Normalized confusion matrix" )
    else :
        print( 'Confusion matrix, with out normalization')
print(cm)

#%% 7
from sklearn.model_selection import cross_val_score
scores = cross_val_score(clftree,d_train_att,d_train_label, cv=5)
skor_rata2=scores.mean()
skoresd=scores.std()
print(skor_rata2)
print(skoresd)

#%% 8
from sklearn.ensemble import RandomForestClassifier
import numpy as np
max_features_opts = range(5 ,50 ,5)
n_estimators_opts = range (10 ,200 ,20)
rf_params = np.empty((len(max_features_opts)*len(n_estimators_opts) ,4),float)
i = 0
for max_features in max_features_opts:
    for n_estimators in n_estimators_opts:
        clf = RandomForestClassifier(max_features=max_features, n_estimators=n_estimators)
        scores = cross_val_score(clf,d_train_att,d_train_label, cv=5)
        rf_params[i,0] = max_features
        rf_params[i,1] = n_estimators
        rf_params[i,2] = scores.mean()
        rf_params[i,3] = scores.std() * 2
        i += 1
        print("Max_features: %d , num estimators: %d , accuracy: %0.2f(+/− %0.2f)" 
              % (max_features, n_estimators, scores.mean(), scores.std() * 2))
