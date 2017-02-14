#Setting up the environment
#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data, test_classifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.cross_validation import train_test_split

### features_list is a list of strings, each of which is a feature name.
### Task 1: Select what features you'll use.
### The first feature must be "poi".

 
### Load the dictionary containing the / m. 
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
    data = pd.DataFrame.from_dict(data_dict,orient='index', dtype=np.float).drop('email_address',1)

# Remove irrelevant outliers
#data = data[~(data.index == 'TOTAL')]
print 'There are %i pois out of %i samples.' %(sum(data['poi']==1),len(data['poi']))

# Summary of missing values across features:
print data.isnull().sum(axis = 0)



### Create new features: ‘from_frac’, ‘to_frac', and 'total_money':

data = data.fillna(0)

data['to_frac'] = data["from_poi_to_this_person"]/data["to_messages"]
data['from_frac'] = data["from_this_person_to_poi"]/data["from_messages"]
data['total_money'] = data['total_payments'] + data["total_stock_value"]
data = data.fillna(0)

# Convert the dataset to dictionary for later submission:
my_dataset = data.to_dict(orient = 'index')


### Investigate Variable Importance:

select = SelectKBest(k = 'all')

from sklearn.cross_validation import StratifiedShuffleSplit
labels = data['poi']
features = data.drop('poi',axis = 1)

# SelectKBest with multiple resamplings:
from sklearn.cross_validation import StratifiedShuffleSplit
cv = StratifiedShuffleSplit(labels, 1500, random_state = 42)
score = []
pval = []
for train_idx, test_idx in cv: 
    features_train = features.iloc[train_idx,:]
    labels_train = labels[train_idx]
    select.fit(features_train,labels_train)
    score.append(select.scores_)
    pval.append(select.pvalues_)
    
avgscore = np.mean(score, axis = 0)
avgpval = np.mean(pval, axis = 0)


print "Variable Importance Chart"
print pd.DataFrame(
    {'Feature': features.columns,
     'Score': [ '%.2f' % elem for elem in avgscore],
     'P-value': [ '%.2f' % elem for elem in avgpval]
    })

indices = np.argsort(avgscore)

plt.figure()
plt.title("Feature importances")
plt.bar(range(len(avgscore)), avgscore[indices],
       color="r", align="center")
plt.xticks(range(len(avgscore)), features.columns[indices],rotation=90)
plt.xlim([-1, len(avgscore)])
plt.show()



# Sets of features to train on:
feature_list1 = ['to_messages','shared_receipt_with_poi','loan_advances','from_this_person_to_poi',
             'from_poi_to_this_person','to_frac','from_frac'] # features with p-val < 0.3
feature_list2 = ['from_frac','shared_receipt_with_poi','exercised_stock_options']
feature_list21 = feature_list2 + ['total_money'] # adding 'total_money'
feature_list22 = feature_list2 + ['to_frac'] # adding 'to_frac'
feature_list23 = ['shared_receipt_with_poi','exercised_stock_options'] # removing 'from_frac'


feature_use = feature_list2 #Replace feature set to run on

features = data[feature_use]
features_list = ['poi'] + feature_use


from sklearn.neighbors import KNeighborsClassifier
clf_kn = KNeighborsClassifier()
from sklearn.linear_model import LogisticRegression
clf_log = LogisticRegression(class_weight = 'balanced',random_state = 111)
from sklearn.ensemble import RandomForestClassifier
clf_rf = RandomForestClassifier(class_weight = 'balanced',random_state = 111)


from sklearn.preprocessing import MinMaxScaler
from sklearn.grid_search import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
pca = PCA()
n_compnt = range(len(feature_use)) + [3]
n_compnt.remove(0)
    
    
pipe_kn = Pipeline(steps = [('pca',pca),('scale',MinMaxScaler()),('kn',clf_kn)])
param_grid_kn = dict(pca__n_components = n_compnt,
                     kn__n_neighbors = [3,5,7,11],
                     kn__weights = ['uniform','distance'])
grid_search_kn = GridSearchCV(pipe_kn, param_grid_kn, scoring = 'recall')
grid_search_kn.fit(features,labels)
test_classifier(grid_search_kn.best_estimator_,my_dataset,features_list)


pipe_log = Pipeline(steps = [('pca',pca),('clf',clf_log)])
param_grid_log = dict(pca__n_components = n_compnt,
                      clf__C = [0.0003,0.001,0.003,0.01,0.03,0.1,0.3,1,3])
grid_search_log = GridSearchCV(pipe_log,param_grid_log,scoring = 'recall')
grid_search_log.fit(features,labels)
test_classifier(grid_search_log.best_estimator_,my_dataset,features_list)

pipe_rf = Pipeline(steps = [('pca',pca),('clf',clf_rf)])
param_grid_rf = dict(pca__n_components = n_compnt,
                     clf__max_depth = [1,2,3,5,7])
grid_search_rf = GridSearchCV(pipe_rf,param_grid_rf,scoring = 'recall')
grid_search_rf.fit(features,labels)
test_classifier(grid_search_rf.best_estimator_,my_dataset,features_list)


clf = grid_search_log.best_estimator_
dump_classifier_and_data(clf, my_dataset, features_list)