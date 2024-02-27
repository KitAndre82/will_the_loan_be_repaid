#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# load file with description of data in columns
data_info = pd.read_csv('/Users/andrew/Downloads/TensorFlow_FILES/DATA/lending_club_info.csv',
                 index_col='LoanStatNew')

# display description of features
print(data_info['Description'])

def feat_info(col_name):
    '''Enter name of column as parameter to get its description'''
    print(data_info.loc[col_name]['Description'])

# load dataset
df = pd.read_csv('/Users/andrew/Downloads/TensorFlow_FILES/DATA/lending_club_loan_two.csv')

## EDA

df.info()

df.duplicated().sum()

df.isnull().sum()

df['emp_title'].unique()

df['emp_title'].nunique()

df['emp_title'].value_counts()

df['title'].unique()

df['title'].nunique()

df['title'].value_counts()

# filling null values
df['emp_title'] = df['emp_title'].fillna('Not given')
df['title'] = df['title'].fillna('Not given')
df['revol_util'] = df['revol_util'].fillna(df['revol_util'].median())
df['mort_acc'] = df['mort_acc'].fillna(df['mort_acc'].median())
df['pub_rec_bankruptcies'] = df['pub_rec_bankruptcies'].fillna(df['pub_rec_bankruptcies'].median())



# converting employment length to numerical values
df['emp_length'] = df['emp_length'].map({'1 year':1,'2 years':2,'3 years':3,'4 years':4,'5 years':5,'6 years':6,
                                        '7 years':7,'8 years':8,'9 years':9,'10+ years':10,'< 1 year':0.5})

# filling null employment length values with median employment length
df['emp_length'] = df['emp_length'].fillna(df['emp_length'].median())

## Feature engineering

# creating title count feature
df['title'] = df['title'].apply(lambda x: x.lower() if isinstance(x,str) else x)

title_dict = dict(df['title'].value_counts())

title_freq = []
for title in df['emp_title']:
    try:
        freq = float(title_dict[title])
        emp_title_freq.append(freq)
    except:
        freq = 0
        title_freq.append(freq)

df['title_freq'] = title_freq

#creating employment title frequency feature
df['emp_title'] = df['emp_title'].apply(lambda x: x.lower() if isinstance(x,str) else x)

emp_title_dict = dict(df['emp_title'].value_counts())

emp_title_dict['not given']

emp_title_freq = []
for title in df['emp_title']:
    try:
        freq = float(emp_title_dict[title])
        emp_title_freq.append(freq)
    except:
        freq = 0
        emp_title_freq.append(freq)

df['emp_title_freq'] = emp_title_freq



df['sub_grade']

# transforming sub grade into numerical value 
df['sub_grade'] = df['sub_grade'].apply(lambda x: int(x.strip(x[:1])))

df['application_type'].unique()

# one hot encoding application type feature
df_app_type = pd.get_dummies(df['application_type'],drop_first=True,prefix='application')

df = df.join(df_app_type).drop('application_type',axis=1)

# one hot encoding purpose feature
df_purpose = pd.get_dummies(df['purpose'],drop_first=True,prefix='purpose')

df = df.join(df_purpose).drop(columns=['purpose'])

# one hot encoding verification status feature
df_source = pd.get_dummies(df['verification_status'],drop_first=True,prefix='verification')

df = df.join(df_source).drop(columns=['verification_status'])

# one hot encoding home ownership feature
df_home_ownership = pd.get_dummies(df['home_ownership'],drop_first=True,prefix='ownership')

df = df.join(df_home_ownership).drop(columns=['home_ownership'])

# one hot encoding initial list status feature
df_init_list = pd.get_dummies(df['initial_list_status'],drop_first=True,prefix='init_list')
df = df.join(df_init_list).drop(columns=['initial_list_status'])

df['address'].loc[396028]


#extracting zip code from address and creating new feature called ZIP_code
df['ZIP_code'] = df['address'].apply(lambda x: int(x[-5:]))

# one hot encoding ZIP_code feature
df_zip = pd.get_dummies(df['ZIP_code'],prefix='Zip',drop_first=True)

df = df.join(df_zip).drop(columns=['ZIP_code'])

# validating dataset after initial feature transformation
df.head(2)

# transforming grade and term feature to numerical feature
df['grade'] = df['grade'].map({'A':1,'B':2,'C':3,'D':4,'E':5,'F':6,'G':7})
df['term'] = df['term'].map({' 36 months':36,' 60 months':60})

# transforming date variables to datetime format
df['issue_d'] = pd.to_datetime(df['issue_d'])

df['earliest_cr_line'] = pd.to_datetime(df['earliest_cr_line'])

# creating borrowing experience feature from datetime features
df['borrowing_experience'] = df['issue_d'] - df['earliest_cr_line']

import datetime as dt
df['borrowing_experience'] = df['borrowing_experience'].dt.days

# feature selection
df_v2 = df.drop(columns=['emp_title','issue_d','title','earliest_cr_line','address'])

#plt.figure(figsize=(10,6))
#sns.heatmap(df.corr())

df_v2.isnull().sum()

# validating dataset
df_v2.head(2)

sns.countplot(x='loan_status',data=df_v2)
plt.title('Class imbalance')
plt.show()

df_v2['loan_status'].value_counts()/len(df)

#sns.scatterplot(y='installment',x='loan_amnt',data=df)

# Transforming outcome variable into binary
df_v2['loan_status'] = df_v2['loan_status'].map({'Charged Off':1,'Fully Paid':0})


# In[2]:


## Building Artificial Neural Network

# Getting sample for model building
df_model = df_v2.sample(int(len(df)/10))

df_model.shape


df_model.head()

# Identifying and assigning independent and target variables
X = df_model.drop(columns=['loan_status']).values
y = df_model['loan_status'].values


# splitting data into training and test data
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.3, stratify=y)

# splitting train data further to include validationvset
X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size = 0.3, stratify=y_train)

# Transforming validation data before using it to train model
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_tr = scaler.fit_transform(X_tr)

X_val = scaler.transform(X_val)

# importing deep learning packages 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# building model
model = Sequential()

model.add(Dense(32,activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(16,activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam')


get_ipython().run_cell_magic('time', '', "# training the model\n\nfrom tensorflow.keras.callbacks import EarlyStopping\n\nearly_stop = EarlyStopping(monitor='val_loss',mode='min',verbose=1,patience=10)\n\nmodel.fit(x=X_tr,y=y_tr,epochs=50,validation_data=(X_val,y_val),\n         callbacks=[early_stop])\n")

# creating loss dataframe
losses = pd.DataFrame(model.history.history)


# plotting loss
losses.plot()

# making predictions of validation data
predictions = model.predict(X_val)

predictions = np.round(predictions).astype(int)


# Testing model
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_val,predictions))
print(confusion_matrix(y_val,predictions))

from sklearn.metrics import recall_score,accuracy_score,precision_score,f1_score

print('Accuracy score: ',accuracy_score(y_val,predictions))
print('Recall score: ',recall_score(y_val,predictions))
print('Precision score: ',precision_score(y_val,predictions))
print('F1 score: ',f1_score(y_val,predictions))


## Evaluating mode with test dataset

X_test = scaler.transform(X_test)

Y_PREDS = model.predict(X_test)

Y_PREDS = np.round(Y_PREDS).astype(int)

ann_classification = classification_report(y_test, Y_PREDS)

print(ann_classification)

ann_confusion = confusion_matrix(y_test, Y_PREDS)

print(ann_confusion)

# Testing artificial neural network model with 30% of original dataset

X_ = df_v2.drop(columns=['loan_status'])
y_ = df_v2['loan_status']

X_TR,X_TEST,y_TR,y_TEST = train_test_split(X_,y_,test_size=0.3,stratify=y_)

X_TEST = scaler.transform(X_TEST)

ann_preds = model.predict(X_TEST)

ann_preds = np.round(ann_preds)

print(confusion_matrix(y_TEST, ann_preds))
print('\n')
print(classification_report(y_TEST, ann_preds))


# ## Decision Tree Classifier

get_ipython().run_cell_magic('time', '', '#Training our model\nfrom sklearn.tree import DecisionTreeClassifier\n\ndtree = DecisionTreeClassifier()\n\ndtree.fit(X_TR, y_TR)\n\npredictions = dtree.predict(X_TEST)\n\n#Evaluating accuracy of model\nfrom sklearn.metrics import classification_report,confusion_matrix\n\ndtree_confusion = confusion_matrix(y_TEST, predictions)\ndtree_report = classification_report(y_TEST, predictions)\nprint(dtree_confusion)\nprint(\'\\n\')\nprint(dtree_report)\n\nfrom sklearn import metrics\nprint("Decision Tree model accuracy:", metrics.accuracy_score(y_TEST, predictions))\nprint("Decision Tree model recall:", metrics.recall_score(y_TEST, predictions))\nprint("Decision Tree model precision:", metrics.precision_score(y_TEST, predictions))\nprint("Decision Tree model F1:", metrics.f1_score(y_TEST, predictions))\n')


# plotting decision tree

from sklearn.tree import plot_tree
plot_tree(dtree, max_depth=3, fontsize=8, feature_names=list(df_model.columns));


# ## Random Forest Classifier


get_ipython().run_cell_magic('time', '', '#Training our model\nfrom sklearn.ensemble import RandomForestClassifier\n\nrfc = RandomForestClassifier()\n\nrfc.fit(X_TR, y_TR)\n\npredictions = dtree.predict(X_TEST)\n\n#Evaluating accuracy of model\nfrom sklearn.metrics import classification_report,confusion_matrix\n\nrfc_confusion = confusion_matrix(y_TEST, predictions)\nrfc_report = classification_report(y_TEST, predictions)\n\nprint(rfc_confusion)\nprint(\'\\n\')\nprint(rfc_report)\n\nfrom sklearn import metrics\nprint("RFC model accuracy:", metrics.accuracy_score(y_TEST, predictions))\nprint("RFC Tree model recall:", metrics.recall_score(y_TEST, predictions))\nprint("RFC Tree model precision:", metrics.precision_score(y_TEST, predictions))\nprint("RFC Tree model F1:", metrics.f1_score(y_TEST, predictions))\n')


# ## Tuned Random Forest Classifier

from sklearn.model_selection import train_test_split, PredefinedSplit, GridSearchCV
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

X_ = df_v2.drop(columns=['loan_status'])
y_ = df_v2['loan_status']

Xtrain, Xtest, ytrain, ytest = train_test_split(X_, y_, test_size = 0.3, stratify=y_)

X_tr, X_val, y_tr, y_val = train_test_split(Xtrain, ytrain, test_size = 0.3, stratify=ytrain)

#Using the Random Forest Classifier algorithm
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier()

# Determine set of hyperparameters.

cv_params = {'n_estimators' : [50,200], 
              'max_depth' : [5,50],        
              'min_samples_leaf' : [0.5,1], 
              'min_samples_split' : [0.001, 0.01],
              'max_features' : ["sqrt"], 
              'max_samples' : [.5,.9]}

# Create list of split indices.

split_index = [0 if x in X_val.index else -1 for x in Xtrain.index]
custom_split = PredefinedSplit(split_index)

# Search over specified parameters.

rfc_val = GridSearchCV(rfc, cv_params, cv=custom_split, refit='f1', n_jobs = -1, verbose = 1)

get_ipython().run_cell_magic('time', '', '\n# Fit the model.\n\nrfc_val.fit(Xtrain,ytrain)\n\n# Obtain optimal parameters.\n\n### YOUR CODE HERE ###\n\nrfc_val.best_params_\n\n# Use optimal parameters on GridSearchCV.\nrfc_opt = RandomForestClassifier(n_estimators = 50, max_depth = 50, \n                                min_samples_leaf = 1, min_samples_split = 0.001,\n                                max_features="sqrt", max_samples = 0.9, random_state = 0)\n\nrfc_opt.fit(Xtrain, ytrain)\n\nrfc_pred = rfc_opt.predict(Xtest)\n\nrfc_tuned_report = classification_report(ytest, rfc_pred)\nrfc_tuned_confusion = confusion_matrix(ytest, rfc_pred)\n\nprint(rfc_tuned_confusion)\nprint(\'\\n\')\nprint(rfc_tuned_report)\n\nprint("RFC model accuracy:", metrics.accuracy_score(ytest, rfc_pred))\nprint("RFC Tree model recall:", metrics.recall_score(ytest, rfc_pred))\nprint("RFC Tree model precision:", metrics.precision_score(ytest, rfc_pred))\nprint("RFC Tree model F1:", metrics.f1_score(ytest, rfc_pred))\n')

# ## Support Vector Machines

X = df_model.drop(columns=['loan_status'])
y = df_model['loan_status']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.33)


scaler = MinMaxScaler()

scaled_features = scaler.fit_transform(X_train)

X_test = scaler.fit_transform(X_test)

from sklearn.svm import SVC

svm = SVC()

svm.fit(scaled_features,y_train)

predictions = svm.predict(X_test)

print(confusion_matrix(y_test,predictions))
print('\n')
print(classification_report(y_test,predictions))






