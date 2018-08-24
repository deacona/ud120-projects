#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))

# print len(enron_data)
all_people = len(enron_data)
print "{0} people in dataset".format(all_people)

# feature_list = []
# for key in enron_data:
# 	print key
# 	for feature in enron_data[key]:
# 		feature_list.append(feature)

# print len(feature_list)

# def make_unique(seq):
#    # not order preserving
#    set = {}
#    map(set.__setitem__, seq, [])
#    return set.keys()

# print len(make_unique(feature_list))

poi_counter = 0
for person_name in enron_data:
	if enron_data[person_name]["poi"]==1:
		poi_counter += 1

# print poi_counter
print "{0} pois in dataset".format(poi_counter)

# print enron_data["PRENTICE JAMES"]
# print enron_data["COLWELL WESLEY"]
# print enron_data["SKILLING JEFFREY K"]["total_payments"] ## (CFO) 8682716
# print enron_data["LAY KENNETH L"]["total_payments"] ## (Chair)    103559793
# print enron_data["FASTOW ANDREW S"]["total_payments"] ##  (CEO)     2424083

salary_counter = 0
email_counter = 0
payments_counter = 0
poi_non_payments_counter = 0

salary_counter += sum(each["salary"] !='NaN' for each in enron_data.itervalues())
email_counter += sum(each["email_address"] !='NaN' for each in enron_data.itervalues())
payments_counter += sum(each["total_payments"] !='NaN' for each in enron_data.itervalues())
poi_non_payments_counter += sum(each["total_payments"] =='NaN' and each["poi"] == 1 for each in enron_data.itervalues())

print "{0} quantified salaries".format(salary_counter)
print "{0} known email addresses".format(email_counter)
print "{0} unknown total payments, i.e. {1}%".format((all_people - payments_counter), 100 * (all_people - payments_counter)/all_people)
print "{0} unknown poi payments, i.e. {1}%".format(poi_non_payments_counter, 100 * poi_non_payments_counter / all_people)