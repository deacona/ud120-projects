#!/usr/bin/python

import sys
import pickle

import pandas as pd
from pandas.tools.plotting import scatter_matrix
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble.weight_boosting import AdaBoostClassifier
from sklearn.cross_validation import train_test_split
from itertools import compress

pd.set_option('float_format', '{:f}'.format)
pd.set_option('display.max_columns', 30)
# pd.set_option('display.expand_frame_repr', False)

sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

import rubric
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


def select_algorithm(labels, features):
	print("\n\n")
	# Split testing and training data
		# test_size
	# Use different algorithms (2+)
		# Naive Bayes
		# Support Vector Machines
	# Tune parameters (try 3+, use 1+)
		# GridSearchCV
	# Evaluation metric (2+)
		# Precision > 0.3
		# Recall > 0.3

	sizes = [0.1, 0.2, 0.3, 0.4, 0.5]
	algos = [GaussianNB(), SVC(kernel='rbf'), AdaBoostClassifier(n_estimators=10)]

	for size in sizes:
		for algo in algos:
			name = algo.__doc__[:24].strip()
			print("Attempting with test_size={0}, using {1}...".format(size, name))
			features_train, features_test, labels_train, labels_test = \
			    train_test_split(features, labels, test_size=size, random_state=42)

			clf = algo
			clf.fit(features_train, labels_train)

			print("SCORE: {0}".format(clf.score(features_test, labels_test)))

	return clf


### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
# features_list = ['poi','salary'] # You will need to use more features
features_list = ['poi', 'salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees', 'to_messages', 'email_address', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi'] ## from project docs

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
	data_dict = pickle.load(data_file)

# rubric.r01_functionality()
# rubric.r02_usability()
# rubric.r03_data_exploration()
explore_dataset(features_list, data_dict, createviz="initial")
explore_dataset(features_list, data_dict, feature="loan_advances")
explore_dataset(features_list, data_dict, feature="director_fees")
explore_dataset(features_list, data_dict, feature="email_address")

### Task 2: Remove outliers
# rubric.r04_outlier_investigation()
explore_dataset(features_list, data_dict, name="TOTAL")
remove_outliers(data_dict)

### Task 3: Create new feature(s)
# rubric.r05_create_new_features()
features_list = add_features(features_list, data_dict)
### Store to my_dataset for easy export below.
my_dataset = data_dict

features_list, labels, features = select_features(features_list, my_dataset)
explore_dataset(features_list, data_dict, createviz="final")
# rubric.r06_intelligently_select()
rubric.r07_properly_scale()


### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
# from sklearn.naive_bayes import GaussianNB
# clf = GaussianNB()
clf = select_algorithm(labels, features)

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

rubric.r08_pick_an_algorithm()
rubric.r09_discuss_parameter()
rubric.r10_tune_the_algorithm()

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.
rubric.r11_usage_of_evaluation()
rubric.r12_discuss_validation()
rubric.r13_validation_strategy()
rubric.r14_algorithm_performance()

dump_classifier_and_data(clf, my_dataset, features_list)