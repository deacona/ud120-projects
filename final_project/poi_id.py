#!/usr/bin/python

import sys
import pickle

import pandas as pd
from pandas.tools.plotting import scatter_matrix
import numpy as np
import matplotlib.pyplot as plt
from itertools import compress
from time import time

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble.weight_boosting import AdaBoostClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, f1_score
# from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import GridSearchCV
# from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
# from sklearn.model_selection import KFold

pd.set_option('float_format', '{:f}'.format)
pd.set_option('display.max_columns', 30)
# pd.set_option('display.expand_frame_repr', False)

sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data, test_classifier

NOSTATS=True


def explore_dataset(features_list, data_dict, name=None, feature=None, createviz=None):
	if NOSTATS:
		return
	print("\n\n")		
	df = pd.DataFrame.from_dict(data_dict, orient='index')
	df.replace({"NaN": np.nan}, inplace=True)
	df = df[features_list]

	if name:
		print("\n\n")
		print("Name: {0}...".format(name))
		print(df.loc[name])
		return

	if feature:
		print("\n\n")
		print("Feature: {0}...".format(feature))
		print(df[df[feature].notnull()][[feature, "poi"]])
		return

	print("\n\nSHAPE...")
	print("{0} rows and {1} columns".format(df.shape[0], df.shape[1]))
	feature_counter = len(df.columns) - 1 #exclude target variable
	print("{0} features".format(feature_counter))
	print("{0} POIs".format(df[df.poi].shape[0]))
	print("\n\nINFO...")
	df.info()
	print("\n\nDESCRIBE...")
	print(df.describe(include="all"))

	if not createviz:
		return

	print("\n\n")
	num_fields = df.columns.values.tolist()
	try:
		num_fields.remove("email_address")
	except ValueError:
		pass
	try:
		num_fields.remove("poi")
	except ValueError:
		pass
	print("num_fields: {0}".format(num_fields))

	print("\n\n")
	print("Plot histograms for all numeric fields...")
	df.hist(column=num_fields) #, bins=146)
	plt.savefig('histograms_{0}.png'.format(createviz))

	print("\n\n")
	print("Plot scatter matrix for all numeric fields...")
	scatter_matrix(df[num_fields], diagonal="kde")
	plt.savefig('scatter_matrix_{0}.png'.format(createviz))

	print("\n\n")
	print("Plot box plots for all numeric fields by POI...")
	df.boxplot(by="poi")
	plt.savefig('boxplot_{0}.png'.format(createviz))

	plt.show()
	return


def remove_outliers(data_dict):
	print("\n\n")
	print("Removing TOTAL as not a person, just a spreadsheet artefact")
	data_dict.pop("TOTAL", 0 )
	return #data_dict


def add_features(features_list, data_dict):
	print("\n\n")
	def computeFraction( poi_messages, all_messages ):
	    """ given a number messages to/from POI (numerator) 
	        and number of all messages to/from a person (denominator),
	        return the fraction of messages to/from that person
	        that are from/to a POI
	    """
	    fraction = 0.
	    
	    def isNaN(num):
	        return num != num
	        
	    if (not isNaN(float(poi_messages))) and (not isNaN(float(all_messages))):
	        fraction = float(poi_messages) / float(all_messages)

	    return fraction

	for name in data_dict:

	    data_point = data_dict[name]

	    from_poi_to_this_person = data_point["from_poi_to_this_person"]
	    to_messages = data_point["to_messages"]
	    fraction_from_poi = computeFraction( from_poi_to_this_person, to_messages )
	    data_point["fraction_from_poi"] = fraction_from_poi

	    from_this_person_to_poi = data_point["from_this_person_to_poi"]
	    from_messages = data_point["from_messages"]
	    fraction_to_poi = computeFraction( from_this_person_to_poi, from_messages )

	    data_dict[name].update( {"fraction_from_poi":fraction_from_poi} )
	    data_dict[name].update( {"fraction_to_poi":fraction_to_poi} )
	    data_point["fraction_to_poi"] = fraction_to_poi

	    if data_dict[name]["email_address"] == "NaN":
	    	data_dict[name].update( {"has_email_address":0} )
	    else:
	    	data_dict[name].update( {"has_email_address":1} )

	new_features_list = features_list+["fraction_from_poi", "fraction_to_poi", "has_email_address"]
	print("Added new features: fraction_from_poi, fraction_to_poi, has_email_address")
	    
	# print(data_dict)
	return new_features_list


def select_features(features_list, my_dataset):
	print("\n\n")
	print("Removing email_address as not numeric or really a useful feature anymore")
	features_list.remove("email_address")

	### Extract features and labels from dataset for local testing
	data = featureFormat(my_dataset, features_list, sort_keys = True)
	labels, features = targetFeatureSplit(data)

	clf = tree.DecisionTreeClassifier(random_state=1)
	clf.fit(features, labels)

	kbest = 0
	selected_features_list = ["poi"] ## always included
	print("\n\n")
	print("Finding most important features...")
	for n, imp in enumerate(clf.feature_importances_):
		print "{0}, {1}, {2}".format(n, features_list[n+1], imp)
		if imp > 0.0:
			kbest+=1
			selected_features_list.append(features_list[n+1])

	print("Selecting {0} best features...".format(kbest))
	selector = SelectKBest(f_classif, k=kbest)
	selector.fit(features, labels)

	# selector_list = [True]+selector.get_support().tolist() ## so poi always included

	# selected_features_list = list(compress(features_list, selector_list))
	selected_features = selector.transform(features)

	print("Selected {0} features".format(len(selected_features_list)))
	print("Selected features are: {0}".format(selected_features_list))

	return selected_features_list, labels, selected_features


def scale_features(original_features):
	scaler = MinMaxScaler()
	# StandardScaler()
	rescaled_features = scaler.fit_transform(original_features)
	return rescaled_features


def select_algorithm(labels, features): # , my_dataset, features_list):
	print("\n\n")

	sizes = [0.1] #, 0.2, 0.3] #, 0.4, 0.5] ## train_test_split
	# sizes = [10] #5, 10, 15, 20] ## KFold
	algos = [
			{"Classifier": GaussianNB(),
			"ParamGrid": {
			    },
			},
			{"Classifier": SVC(kernel="rbf"),
			"ParamGrid": {
				'C': [1e3, 5e3, 1e4, 5e4, 1e5],
			    'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1],
				},
			},
			{"Classifier": KNeighborsClassifier(),
			"ParamGrid": {
				"n_neighbors": [3, 5, 7],
				"p": [1, 2, 3]
			    },
			},
			{"Classifier": AdaBoostClassifier(),
			"ParamGrid": {
				"n_estimators": [30, 40, 50],
				"learning_rate": [0.5, 1.0, 1.5],
			    },
			},
		]

	best_clf = None
	best_score = 0

	for size in sizes:
		for algo in algos:
			name = algo["Classifier"].__doc__[:24].strip()
			print("\n\n")
			print("Attempting with test_size={0}, using {1}...".format(size, name))
			features_train, features_test, labels_train, labels_test = \
			    train_test_split(features, labels, test_size=size, random_state=42)
			# print("Attempting with kfolds={0}, using {1}...".format(size, name))
			# kf = KFold(n_splits=size, shuffle=True, random_state=42)
			# for train_index, test_index in kf.split(features):
				# features_train = [features[ii] for ii in train_index]
				# features_test = [features[ii] for ii in test_index]
				# labels_train = [labels[ii] for ii in train_index]
				# labels_test = [labels[ii] for ii in test_index]
			# print("Attempting with kfolds={0} in GridSearchCV, using {1}...".format(size, name))
			# features_train = features
			# labels_train = labels
			# features_test = features
			# labels_test = labels

			t0 = time()
			# folds=int(1/size)
			clf = GridSearchCV(algo["Classifier"], algo["ParamGrid"]) #, cv=folds)
			clf.fit(features_train, labels_train)
			print("Done in %0.3fs" % (time() - t0))
			print("Best estimator found by grid search:")
			print(clf.best_estimator_)
			score = clf.score(features_test, labels_test)
			print("Score: {0}".format(score))

			labels_pred = clf.predict(features_test)
			print("Classification report:")
			print(classification_report(labels_test, labels_pred))

			f1 = f1_score(labels_test, labels_pred)
			print("F1 score: {0}".format(f1))

			# test_classifier(clf, my_dataset, features_list)

			# if score > best_score:
			# 	best_score = score
			# 	best_clf = clf.best_estimator_
			if f1 > best_score:
				best_score = f1
				best_clf = clf.best_estimator_

	print("\n\n")				
	print("Best classifier:")
	print(best_clf)
	print("Best score: {0}".format(best_score))
	labels_pred = best_clf.predict(features_test)
	print("Best classification report:")
	print(classification_report(labels_test, labels_pred))

	return best_clf


### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
# features_list = ['poi','salary'] # You will need to use more features
features_list = ['poi', 'salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees', 'to_messages', 'email_address', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi'] ## from project docs

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
	data_dict = pickle.load(data_file)

explore_dataset(features_list, data_dict, createviz="initial")
explore_dataset(features_list, data_dict, feature="loan_advances")
explore_dataset(features_list, data_dict, feature="director_fees")
explore_dataset(features_list, data_dict, feature="email_address")

### Task 2: Remove outliers
explore_dataset(features_list, data_dict, name="TOTAL")
remove_outliers(data_dict)

### Task 3: Create new feature(s)
features_list = add_features(features_list, data_dict)
### Store to my_dataset for easy export below.
my_dataset = data_dict

features_list, labels, features = select_features(features_list, my_dataset)
explore_dataset(features_list, data_dict, createviz="final")

features = scale_features(features)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
# from sklearn.naive_bayes import GaussianNB
# clf = GaussianNB()
clf = select_algorithm(labels, features) #, my_dataset, features_list)

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
# from sklearn.cross_validation import train_test_split
# features_train, features_test, labels_train, labels_test = \
#     train_test_split(features, labels, test_size=0.3, random_state=42)
test_classifier(clf, my_dataset, features_list)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)