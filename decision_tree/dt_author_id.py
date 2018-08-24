#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:    
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

# print len(features_train[0])
# sys.exit()

from sklearn import tree

clf = tree.DecisionTreeClassifier(min_samples_split=40)

# features_train = features_train[:len(features_train)/100]
# labels_train = labels_train[:len(labels_train)/100]

t0 = time()
clf.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"

t1 = time()
pred = clf.predict(features_test)
print "prediction time:", round(time()-t1, 3), "s"

print "SCORE: {0}".format(clf.score(features_test, labels_test))

# print "PREDICTION Chris(1): {0}".format(sum(pred))


## RESULTS

## 10% of features
# training time: 5.015 s
# prediction time: 0.004 s
# SCORE: 0.967007963595

## 1% of features
# training time: 5.015 s
# prediction time: 0.004 s
# SCORE: 0.967007963595



#########################################################


