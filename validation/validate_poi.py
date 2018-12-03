#!/usr/bin/python


"""
    Starter code for the validation mini-project.
    The first step toward building your POI identifier!

    Start by loading/formatting the data

    After that, it's not our code anymore--it's yours!
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### first element is our labels, any added elements are predictor
### features. Keep this the same for the mini-project, but you'll
### have a different feature list when you do the final project.
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)



### it's all yours from here forward!  

from sklearn import tree
from sklearn import cross_validation
from sklearn.metrics import confusion_matrix, precision_score, recall_score
import numpy as np

features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(features, labels, test_size=0.3, random_state=42)

clf = tree.DecisionTreeClassifier()

# clf.fit(features, labels)
clf.fit(features_train, labels_train)

# pred = clf.predict(features)
pred = clf.predict(features_test)

# pred = array([0.0] * 29)
# pred = np.full(
#     shape=29,
#     fill_value=0.0,
#     dtype=np.float)

# print "Labels: {0}".format(labels_test)

print "pred: {0}".format(pred)

# print "SCORE: {0}".format(clf.score(features, labels))
print "SCORE: {0}".format(clf.score(features_test, labels_test))

print "PRECISION: {0}".format(precision_score(labels_test, pred))
print "RECALL: {0}".format(recall_score(labels_test, pred))

print confusion_matrix(labels_test, pred, labels=range(2))

