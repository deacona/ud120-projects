#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###

from sklearn.svm import SVC

# clf = SVC(kernel='linear')
# clf = SVC(kernel='rbf')
clf = SVC(kernel='rbf', C=10000.)

# features_train = features_train[:len(features_train)/100]
# labels_train = labels_train[:len(labels_train)/100]

t0 = time()
clf.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"

t1 = time()
pred = clf.predict(features_test)
print "prediction time:", round(time()-t1, 3), "s"

print "SCORE: {0}".format(clf.score(features_test, labels_test))

print "PREDICTION Chris(1): {0}".format(sum(pred))

# for i in range(1,10):
# for i in [10, 26, 50]:
# 	pred = clf.predict(features_test[i].reshape(1, -1))
# 	print "\nEXAMPLE: {0} - PREDICTION: {2} vs ACTUAL: {3}".format(i, features_test[i], pred, labels_test[i])


### RESULTS

# training time: 174.55 s
# prediction time: 20.9 s
# SCORE: 0.984072810011

## 1% filter
# training time: 0.11 s
# prediction time: 1.078 s
# SCORE: 0.884527872582

## 1% filter/rbf
# training time: 0.11 s
# prediction time: 1.344 s
# SCORE: 0.616040955631

## 1% filter/rbf/C=10
# training time: 0.11 s
# prediction time: 1.171 s
# SCORE: 0.616040955631

## 1% filter/rbf/C=100
# training time: 0.11 s
# prediction time: 1.218 s
# SCORE: 0.616040955631

## 1% filter/rbf/C=1000
# training time: 0.109 s
# prediction time: 1.125 s
# SCORE: 0.821387940842

## 1% filter/rbf/C=10000
# training time: 0.109 s
# prediction time: 0.969 s
# SCORE: 0.892491467577
# EXAMPLE: 10 - PREDICTION: [1] vs ACTUAL: 1
# EXAMPLE: 26 - PREDICTION: [0] vs ACTUAL: 0
# EXAMPLE: 50 - PREDICTION: [1] vs ACTUAL: 1

## rbf/C=10000
# training time: 117.734 s
# prediction time: 11.391 s
# SCORE: 0.990898748578
# PREDICTION Chris(1): 877

#########################################################


