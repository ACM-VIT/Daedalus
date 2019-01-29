#!/usr/bin/python

import sys
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data,test_classifier
from sklearn.pipeline import Pipeline,FeatureUnion
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import IsolationForest,RandomForestClassifier,AdaBoostClassifier
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.neighbors import LocalOutlierFactor,NearestNeighbors,KNeighborsClassifier
from sklearn.metrics import recall_score,precision_score,accuracy_score,f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split
from sklearn.decomposition import PCA
### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi', 'salary', 'bonus', 'long_term_incentive',
                 'other', 'expenses','total_payments',
                 'exercised_stock_options', 'restricted_stock', 
                 'total_stock_value', 'to_messages', 'from_messages', 'from_this_person_to_poi', 
                 'from_poi_to_this_person', 'shared_receipt_with_poi'] # You will need to use more features
### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
##convert to pandas dataframe
df=pd.DataFrame.from_dict(data_dict,orient='index')
df.replace('NaN',0.0,inplace= True)
### Task 3: Create new feature(s)
df['fraction_from_poi'] = df['from_poi_to_this_person'] / df['to_messages']
df['fraction_to_poi'] = df['from_this_person_to_poi'] / df['from_messages']
df['bonus_to_salary'] = df['bonus']/df['salary']
data_dict = df.fillna(value='NaN').to_dict(orient='index')
features_list+=['fraction_from_poi','fraction_to_poi','bonus_to_salary']
### Task 2: Remove outliers
##def deloutliers(dictionary,data,contamination=0.02):
##    out=LocalOutlierFactor(contamination=contamination)
##    labels, features = targetFeatureSplit(data)
##    for key,val in zip(dictionary.keys(),out.fit_predict(features,labels)):
##        if val==-1:
##            del dictionary[key]
##    return dictionary

data_dict.pop('TOTAL',0)
data_dict.pop('THE TRAVEL AGENCY IN THE PARK')
data_dict.pop('LOCKHART EUGENE E')


##data_dict = deloutliers(data_dict,featureFormat(data_dict, features_list,sort_keys = True))

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(data_dict, features_list,sort_keys = True)
labels, features = targetFeatureSplit(data)



### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!

############# Task 4 and Task 5 ####################
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)
####  creating pipeline using PCA and standard scaler(Good result) #####
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
pca_svm = Pipeline([('pca',PCA()),('scaler',StandardScaler()),('svm',SVC())]) # creating a pipeline for combining multiple processes
param_grid = ([{'pca__n_components': list(range(1,len(features_list))),
                'svm__C': [1000,10000],
                'svm__gamma': [0.01,0.0001],
                'svm__degree':[2,3],
                'svm__kernel': ['linear','rbf','poly']}])
svm_clf = GridSearchCV(pca_svm,param_grid,scoring='precision').fit(features_train,labels_train).best_estimator_
pca_knb = Pipeline([('pca',PCA()),('scaler',StandardScaler()),('knb',KNeighborsClassifier())])
param_grid = ([{'pca__n_components': list(range(1,len(features_list))),
            'knb__n_neighbors': [4,5,6]}])
knb_clf = GridSearchCV(pca_knb,param_grid,scoring='precision').fit(features_train,labels_train).best_estimator_
pca_rfst = Pipeline([('pca',PCA()),('scaler',StandardScaler()),
                 ('rfst',RandomForestClassifier())])
param_grid = ([{'pca__n_components': list(range(1,len(features_list))),
                'rfst__n_estimators': [4,5,6]}])
rfst_clf = GridSearchCV(pca_rfst,param_grid,scoring='precision').fit(features_train,labels_train).best_estimator_
pca_tree = Pipeline([('pca',PCA()),('scaler',StandardScaler()),('tree',DecisionTreeClassifier())])
param_grid = ([{'pca__n_components': list(range(1,len(features_list))),
                'tree__criterion':['gini','entropy'],
                'tree__min_samples_split' :[2,4,6,8,10,20],
                'tree__max_features' : [None,'sqrt','log2','auto']}])
tree_clf = GridSearchCV(pca_tree,param_grid,scoring='precision').fit(features_train,labels_train).best_estimator_
pca_nb = Pipeline([('pca',PCA()),('scaler',StandardScaler()),('nb',GaussianNB())])
param_grid = ([{'pca__n_components': list(range(1,len(features_list)))}])
nb_clf = GridSearchCV(pca_nb,param_grid,scoring='precision').fit(features_train,labels_train).best_estimator_
print svm_clf
test_classifier(svm_clf,my_dataset,features_list)
print knb_clf
test_classifier(knb_clf,my_dataset,features_list)
print rfst_clf
test_classifier(rfst_clf,my_dataset,features_list)
print tree_clf
test_classifier(tree_clf,my_dataset,features_list)
print nb_clf
test_classifier(nb_clf,my_dataset,features_list)

####  creating pipeline using SelectKbest and standard scaler(medium result) #####
'''
eng_svm = Pipeline([('scaler',StandardScaler()),('kbest',SelectKBest()),('svm',SVC())])
param_grid = ([{'kbest__k':list(range(1,len(features_list))),
                'svm__C': [1,10,100,1000],
                'svm__gamma': [1,0.1,0.01,0.001],
                'svm__degree':[2,3,4],
                'svm__kernel': ['linear','rbf','poly']}])
svm_clf = GridSearchCV(eng_svm,param_grid,scoring='recall').fit(features,labels).best_estimator_
eng_knb = Pipeline([('scaler',StandardScaler()),('kbest',SelectKBest()),('knb',KNeighborsClassifier())])
param_grid = ([{'kbest__k':list(range(1,len(features_list))),'knb__n_neighbors': [2,3,4,5,6]}])
knb_clf = GridSearchCV(eng_knb,param_grid,scoring='recall').fit(features,labels).best_estimator_
eng_rfst = Pipeline([('scaler',StandardScaler()),('kbest',SelectKBest()),
                 ('rfst',RandomForestClassifier())])
param_grid = ([{'kbest__k':list(range(1,len(features_list))),'rfst__n_estimators': [2,3,4,5,6]}])
rfst_clf = GridSearchCV(eng_rfst,param_grid,scoring='recall').fit(features,labels).best_estimator_
eng_tree = Pipeline([('kbest',SelectKBest()),('scaler',StandardScaler()),('tree',DecisionTreeClassifier())])
param_grid = ([{'kbest__k':list(range(1,len(features_list))),
                'tree__criterion':['gini','entropy'],
                'tree__min_samples_split' :[2,4,6,8,10,20],
                'tree__max_features' : [None,'sqrt','log2','auto']}])
tree_clf = GridSearchCV(eng_tree,param_grid,scoring='recall').fit(features,labels).best_estimator_
print svm_clf
test_classifier(svm_clf,my_dataset,features_list)
print knb_clf
test_classifier(knb_clf,my_dataset,features_list)
print rfst_clf
test_classifier(rfst_clf,my_dataset,features_list)
print tree_clf
test_classifier(tree_clf,my_dataset,features_list)
'''
#### combining both SelectKbest and PCA using Feature union(poor result) ######
'''
combined_features = FeatureUnion([("pca", PCA()), ("kbest", SelectKBest())])

hybrid_svm = Pipeline([('features',combined_features),('scaler',StandardScaler()),('svm',SVC())])
param_grid = ([{'features__pca__n_components':list(range(1,len(features_list))),'features__kbest__k':list(range(1,len(features_list))),
                'svm__C': [1,10,100,1000],
                'svm__gamma': [1,0.1,0.01,0.001],
                'svm__degree':[2,3,4],
                'svm__kernel': ['rbf','poly']}])
svm_clf = GridSearchCV(hybrid_svm,param_grid,scoring='recall').fit(features,labels).best_estimator_
hybrid_knb = Pipeline([('features',combined_features),('scaler',StandardScaler()),('knb',KNeighborsClassifier())])
param_grid = ([{'features__pca__n_components':list(range(1,len(features_list))),'features__kbest__k':list(range(1,len(features_list))),
                'knb__n_neighbors': [1,2,3,4,5,6,7]}])
knb_clf = GridSearchCV(hybrid_knb,param_grid,scoring='recall').fit(features,labels).best_estimator_
hybrid_rfst = Pipeline([('features',combined_features),('scaler',StandardScaler()),
                 ('rfst',RandomForestClassifier())])
param_grid = ([{'features__pca__n_components':list(range(1,len(features_list))),'features__kbest__k':list(range(1,len(features_list))),
                'rfst__n_estimators': [2,3,4,5,6,7]}])
rfst_clf = GridSearchCV(hybrid_rfst,param_grid,scoring='recall').fit(features,labels).best_estimator_
hybrid_tree = Pipeline([('features',combined_features),('scaler',StandardScaler()),('tree',DecisionTreeClassifier())])
param_grid =([{'features__pca__n_components':list(range(1,len(features_list))),
               'features__kbest__k' : list(range(1,len(features_list))),
               'tree__criterion':['gini','entropy'],
               'tree__max_features' : [None,'sqrt','log2','auto']}])
tree_clf = GridSearchCV(hybrid_tree,param_grid,scoring='recall').fit(features,labels).best_estimator_
print svm_clf
test_classifier(svm_clf,my_dataset,features_list)
print knb_clf
test_classifier(knb_clf,my_dataset,features_list)
print rfst_clf
test_classifier(rfst_clf,my_dataset,features_list)
print tree_clf
test_classifier(tree_clf,my_dataset,features_list)'''





### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

#dump_classifier_and_data(clf, my_dataset, features_list)

