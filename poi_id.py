#!/usr/bin/python

import sys
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
import tester
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
from sklearn.metrics import recall_score, precision_score
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data



with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
    
enron = pd.DataFrame.from_dict(data_dict, orient = 'index')

enron.replace(to_replace='NaN', value=np.nan, inplace=True)


enron = enron.drop(['TOTAL'])
enron = enron.drop(['THE TRAVEL AGENCY IN THE PARK'])

def frac(x):
    poi_messages = x[0]
    all_messages = x[1]
    fraction = float(poi_messages)/float(all_messages)

    return fraction


email_features = ['to_messages', 'from_messages', 'from_this_person_to_poi', 'from_poi_to_this_person',
                  'shared_receipt_with_poi']

def impute_median(series):
    return series.fillna(series.median())

by_poi = enron.groupby(['poi'])

enron[email_features] = by_poi[email_features].transform(impute_median)

enron["fraction_from_poi"] = enron[["from_poi_to_this_person","to_messages"]].apply(frac, axis = 1)
enron["fraction_to_poi"] = enron[["from_this_person_to_poi","from_messages"]].apply(frac, axis = 1)
enron["fraction_shared_poi"] = enron[["shared_receipt_with_poi","to_messages"]].apply(frac, axis = 1)


features_list = ['poi','salary', 'deferral_payments', 'loan_advances', 'bonus', 
                     'restricted_stock_deferred', 'deferred_income', 'expenses', 'exercised_stock_options', 
                     'other', 'long_term_incentive', 'restricted_stock', 'director_fees', 
                     "fraction_shared_poi","fraction_from_poi","fraction_to_poi"] 


enron.replace(to_replace=np.nan, value='NaN', inplace=True)
my_dataset = enron.to_dict('index')

data = featureFormat(my_dataset, features_list, remove_NaN=True, sort_keys = True)
labels, features = targetFeatureSplit(data)

clf= Pipeline([("preprocessing",SelectKBest(k='all')),
               ("classifier",DecisionTreeClassifier(criterion='entropy', max_depth =2,
                                                    min_samples_split=2, min_samples_leaf=1))])

clf.fit(features, labels)
dump_classifier_and_data(clf, my_dataset, features_list)
tester.main() 