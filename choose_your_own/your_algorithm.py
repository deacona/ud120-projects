#!/usr/bin/python

import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture

features_train, labels_train, features_test, labels_test = makeTerrainData()


### the training data (features_train, labels_train) have both "fast" and "slow"
### points mixed together--separate them so we can give them different colors
### in the scatterplot and identify them visually
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


#### initial visualization
# plt.xlim(0.0, 1.0)
# plt.ylim(0.0, 1.0)
# plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
# plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
# plt.legend()
# plt.xlabel("bumpiness")
# plt.ylabel("grade")
# plt.show()
################################################################################


### your code here!  name your classifier object clf if you want the 
### visualization code (prettyPicture) to show you the decision boundary

from sklearn import ensemble
import sys
from time import time

# clf = ensemble.weight_boosting.AdaBoostClassifier()
# clf = ensemble.weight_boosting.AdaBoostClassifier(algorithm="SAMME")
# clf = ensemble.weight_boosting.AdaBoostClassifier(n_estimators=40)
# clf = ensemble.weight_boosting.AdaBoostClassifier(n_estimators=60)
# clf = ensemble.weight_boosting.AdaBoostClassifier(n_estimators=200)
clf = ensemble.weight_boosting.AdaBoostClassifier(n_estimators=10)
# clf = ensemble.weight_boosting.AdaBoostClassifier(learning_rate=0.5)
# clf = ensemble.weight_boosting.AdaBoostClassifier(learning_rate=1.5)
# clf = ensemble.weight_boosting.AdaBoostClassifier(random_state=1)

t0 = time()
clf.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"

t1 = time()
pred = clf.predict(features_test)
print "prediction time:", round(time()-t1, 3), "s"

print "SCORE: {0}".format(clf.score(features_test, labels_test))


## RESULTS

## out-of-the-box
# training time: 0.125 s
# prediction time: 0.0 s
# SCORE: 0.924

## algorithm="SAMME"
# training time: 0.141 s
# prediction time: 0.015 s
# SCORE: 0.924

## n_estimators=40
# training time: 0.125 s
# prediction time: 0.0 s
# SCORE: 0.924

## n_estimators=60
# training time: 0.157 s
# prediction time: 0.015 s
# SCORE: 0.924

## n_estimators=200
# training time: 0.5 s
# prediction time: 0.031 s
# SCORE: 0.916

## n_estimators=10
# training time: 0.031 s
# prediction time: 0.0 s
# SCORE: 0.916

## learning_rate=0.5
# training time: 0.125 s
# prediction time: 0.015 s

## learning_rate=1.5
# training time: 0.125 s
# prediction time: 0.0 s
# SCORE: 0.924

## random_state=1
# training time: 0.14 s
# prediction time: 0.0 s
# SCORE: 0.924

try:
    prettyPicture(clf, features_test, labels_test)
except NameError:
    pass
