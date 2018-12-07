import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pd.set_option('float_format', '{:f}'.format)
pd.set_option('display.max_columns', 30)
# pd.set_option('display.expand_frame_repr', False)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.DEBUG)

def r01_functionality():
	"""
	Quality of Code - Functionality
	-------------------------------
	Code reflects the description in the answers to questions in the writeup. i.e. code performs the functions documented in the writeup and the writeup clearly specifies the final analysis strategy.
	"""
	logging.info(r01_functionality.__doc__)

	return


def r02_usability():
	"""
	Quality of Code - Usability
	-------------------------------
	poi_id.py can be run to export the dataset, list of features and algorithm, so that the final algorithm can be checked easily using tester.py.
	"""
	logging.info(r02_usability.__doc__)

	return


def r03_data_exploration(data_dict):
	"""
	Understanding the Dataset and Question - Data Exploration (related lesson: Datasets and Questions)
	-------------------------------
	Student response addresses the most important characteristics of the dataset and uses these characteristics to inform their analysis. Important characteristics include:

	    total number of data points
	    allocation across classes (POI/non-POI)
	    number of features used
	    are there features with many missing values? etc.

	"""
	logging.info(r03_data_exploration.__doc__)

	logging.debug("{0} people in dataset".format(len(data_dict)))

	poi_counter = 0
	missing_counter_all = 0
	for person_name in data_dict:
		if data_dict[person_name]["poi"]==1:
			poi_counter += 1
		missing_counter = 0
		for feature_name in data_dict[person_name]:
			if data_dict[person_name][feature_name] == "NaN":
				missing_counter += 1
		# logging.debug("{0} has {1} missing features".format(person_name, missing_counter))
		missing_counter_all += missing_counter
	logging.debug("Overall there are {0} missing features".format(missing_counter_all))

	logging.debug("{0} pois in dataset".format(poi_counter))

	# logging.info(sum(len(x) for x in data_dict.values()))
	df = pd.DataFrame.from_dict(data_dict, orient='index')
	df.replace({"NaN": np.nan}, inplace=True)
	df.info()
	print df.describe()
	# print df.head(5)
	# print df.shape
	feature_counter = len(df.columns) - 1 #exclude target variable
	logging.debug("{0} features in dataset".format(feature_counter))
	# df.info()

	logging.debug("Info on Jeff Skilling...\n{0}".format(data_dict["SKILLING JEFFREY K"]))
	# print df.loc["SKILLING JEFFREY K"]

	for feature in ["loan_advances", "director_fees"]:
		logging.debug("Employees with {0}...\n".format(feature))
		print df[df[feature] > 0][[feature, "poi"]]

	logging.debug("View all email addresses...\n")
	# print df[df["email_address"].notnull()][["email_address", "poi"]]
	print df[df["poi"]][["email_address", "poi"]]

	# num_fields = df.columns.values.tolist()
	# num_fields.remove("email_address")
	# num_fields.remove("poi")
	# logging.debug("num_fields: {0}".format(num_fields))

	# logging.debug("Plot histograms for all numeric fields...")
	# fig = plt.figure()
	# hist = df.hist(column=num_fields, bins=146)
	# plt.show()
	# # fig.savefig('histograms.pdf')

	return


def r04_outlier_investigation():
	"""
	Understanding the Dataset and Question - Outlier Investigation (related lesson: Outliers)
	-------------------------------
	Student response identifies outlier(s) in the financial data, and explains how they are removed or otherwise handled.
	"""
	logging.info(r04_outlier_investigation.__doc__)

	return


def r05_create_new_features(data_dict):
	"""
	Optimize Feature Selection/Engineering - Create new features (related lesson: Feature Selection)
	-------------------------------
	At least one new feature is implemented. Justification for that feature is provided in the written response. The effect of that feature on final algorithm performance is tested or its strength is compared to other features in feature selection. The student is not required to include their new feature in their final feature set.
	"""
	logging.info(r05_create_new_features.__doc__)

	def computeFraction( poi_messages, all_messages ):
	    """ given a number messages to/from POI (numerator) 
	        and number of all messages to/from a person (denominator),
	        return the fraction of messages to/from that person
	        that are from/to a POI
	    """
	    ### you fill in this code, so that it returns either
	    ###     the fraction of all messages to this person that come from POIs
	    ###     or
	    ###     the fraction of all messages from this person that are sent to POIs
	    ### the same code can be used to compute either quantity

	    ### beware of "NaN" when there is no known email address (and so
	    ### no filled email features), and integer division!
	    ### in case of poi_messages or all_messages having "NaN" value, return 0.
	    fraction = 0.
	    
	    def isNaN(num):
	        return num != num
	    # import math
	        
	    if (not isNaN(float(poi_messages))) and (not isNaN(float(all_messages))):
	        fraction = float(poi_messages) / float(all_messages)

	    return fraction


	# submit_dict = {}
	for name in data_dict:

	    data_point = data_dict[name]

	    # print
	    from_poi_to_this_person = data_point["from_poi_to_this_person"]
	    to_messages = data_point["to_messages"]
	    fraction_from_poi = computeFraction( from_poi_to_this_person, to_messages )
	    # print fraction_from_poi
	    data_point["fraction_from_poi"] = fraction_from_poi

	    from_this_person_to_poi = data_point["from_this_person_to_poi"]
	    from_messages = data_point["from_messages"]
	    fraction_to_poi = computeFraction( from_this_person_to_poi, from_messages )
	    # print fraction_to_poi
	    # submit_dict[name]={"from_poi_to_this_person":fraction_from_poi,
	                       # "from_this_person_to_poi":fraction_to_poi}
	    data_dict[name].update( {"from_poi_to_this_person":fraction_from_poi} )
	    data_dict[name].update( {"from_this_person_to_poi":fraction_to_poi} )
	    data_point["fraction_to_poi"] = fraction_to_poi
	    
	# print data_dict
	return data_dict


def r06_intelligently_select():
	"""
	Optimize Feature Selection/Engineering - Intelligently select features (related lesson: Feature Selection)
	-------------------------------
	Univariate or recursive feature selection is deployed, or features are selected by hand (different combinations of features are attempted, and the performance is documented for each one). Features that are selected are reported and the number of features selected is justified. For an algorithm that supports getting the feature importances (e.g. decision tree) or feature scores (e.g. SelectKBest), those are documented as well.
	"""
	logging.info(r06_intelligently_select.__doc__)

	return


def r07_properly_scale():
	"""
	Optimize Feature Selection/Engineering - Properly scale features (related lesson: Feature Scaling)
	-------------------------------
	If algorithm calls for scaled features, feature scaling is deployed.
	"""
	logging.info(r07_properly_scale.__doc__)

	return


def r08_pick_an_algorithm():
	"""
	Pick and Tune an Algorithm - Pick an algorithm (related lessons: Naive Bayes through Choose Your Own Algorithm)
	-------------------------------
	At least two different algorithms are attempted and their performance is compared, with the best performing one used in the final analysis.
	"""
	logging.info(r08_pick_an_algorithm.__doc__)

	return


def r09_discuss_parameter():
	"""
	Pick and Tune an Algorithm - Discuss parameter tuning and its importance.
	-------------------------------
	Response addresses what it means to perform parameter tuning and why it is important.
	"""
	logging.info(r09_discuss_parameter.__doc__)

	return


def r10_tune_the_algorithm():
	"""
	Pick and Tune an Algorithm - Tune the algorithm (related lesson: Validation)
	-------------------------------
	At least one important parameter tuned with at least 3 settings investigated systematically, or any of the following are true:

	    GridSearchCV used for parameter tuning
	    Several parameters tuned
	    Parameter tuning incorporated into algorithm selection (i.e. parameters tuned for more than one algorithm, and best algorithm-tune combination selected for final analysis).

	"""
	logging.info(r10_tune_the_algorithm.__doc__)

	return


def r11_usage_of_evaluation():
	"""
	Validate and Evaluate - Usage of Evaluation Metrics (related lesson: Evaluation Metrics)
	-------------------------------
	At least two appropriate metrics are used to evaluate algorithm performance (e.g. precision and recall), and the student articulates what those metrics measure in context of the project task.
	"""
	logging.info(r11_usage_of_evaluation.__doc__)

	return


def r12_discuss_validation():
	"""
	Validate and Evaluate - Discuss validation and its importance.
	-------------------------------
	Response addresses what validation is and why it is important.
	"""
	logging.info(r12_discuss_validation.__doc__)

	return


def r13_validation_strategy():
	"""
	Validate and Evaluate - Validation Strategy (related lesson Validation)
	-------------------------------
	Performance of the final algorithm selected is assessed by splitting the data into training and testing sets or through the use of cross validation, noting the specific type of validation performed.
	"""
	logging.info(r13_validation_strategy.__doc__)

	return


def r14_algorithm_performance():
	"""
	Validate and Evaluate - Algorithm Performance
	-------------------------------
	When tester.py is used to evaluate performance, precision and recall are both at least 0.3.
	"""
	logging.info(r14_algorithm_performance.__doc__)

	return


