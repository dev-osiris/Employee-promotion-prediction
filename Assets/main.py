# This file produces 'model.pkl' and 'scalar.pkl' which is supposed to be used by employee_pred.py
# for making predicions.

import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

"""
 Reading the Dataset

* Here, we are having two datasets, i.e., Training and Testing Datasets
* We will read both the datasets 
* Training Datasets is used to train the Machine learning Models
* After learning the patterns from the Testing Datasets, We have to predict the Target Variable.
"""


test = pd.read_csv("Assets\\dataset\\test (1).csv")
train = pd.read_csv('Assets\\dataset\\train (1).csv')


# missing values in training data set

# calculate the total missing values in the dataset
train_total = train.isnull().sum()

# calculate the percentage of missing values in the dataset
train_percent = ((train.isnull().sum() / train.shape[0]) * 100).round(2)

# calculate the total missing values in the dataset
test_total = test.isnull().sum()

# calculate the percentage of missing values in the dataset
test_percent = ((test.isnull().sum() / test.shape[0]) * 100).round(2)

# make a dataset consisting of total no. of missing values and percentage of missing values in the dataset
train_missing_data = pd.concat([train_total, train_percent, test_total, test_percent],
                               axis=1,
                               keys=['Train_Total', 'Train_Percent %', 'Test_Total', 'Test_Percent %'],
                               sort=True)


# lets impute NA in the missing values in the Training Data

train['education'] = train['education'].fillna(train['education'].mode()[0])
train['previous_year_rating'] = train['previous_year_rating'].fillna(train['previous_year_rating'].mode()[0])


# impute the missing values in the Testing Data

test['education'] = test['education'].fillna(test['education'].mode()[0])
test['previous_year_rating'] = test['previous_year_rating'].fillna(test['previous_year_rating'].mode()[0])


# lets remove the outliers from the length of service column

train = train[train['length_of_service'] > 13]


# Feature Engineering

# create some extra features from existing features to improve our Model

# creating a Metric of Sum
train['sum_metric'] = train['awards_won?'] + train['KPIs_met >80%'] + train['previous_year_rating']
test['sum_metric'] = test['awards_won?'] + test['KPIs_met >80%'] + test['previous_year_rating']

# creating a total score column
train['total_score'] = train['avg_training_score'] * train['no_of_trainings']
test['total_score'] = test['avg_training_score'] * test['no_of_trainings']

# remove some of the columns which are not very useful for predicting the promotion.

# we already know that the recruitment channel is very least related to promotion of an employee,
# so lets remove this column.
# even the region seems to contribute very less, when it comes to promotion, so lets remove it too.
# also the employee id is not useful so lets remove it.

train = train.drop(['recruitment_channel', 'region', 'employee_id'], axis=1)
test = test.drop(['recruitment_channel', 'region', 'employee_id'], axis=1)

# lets check the columns in train and test data set after feature engineering
# train.columns


# lets remove the above two columns as they have a huge negative effect on our training data

train = train.drop(train[(train['KPIs_met >80%'] == 0) & (train['previous_year_rating'] == 1.0) &
                         (train['awards_won?'] == 0) & (train['avg_training_score'] < 60) & (
                                     train['is_promoted'] == 1)].index)


# Dealing with Categorical Columns

# encoding education column to convert it into numerical column

train['education'] = train['education'].replace(("Master's & above", "Bachelor's", "Below Secondary"),
                                                (3, 2, 1))
test['education'] = test['education'].replace(("Master's & above", "Bachelor's", "Below Secondary"),
                                              (3, 2, 1))


# use Label Encoding for Gender and Department to convert them into Numerical
le = LabelEncoder()
train['department'] = le.fit_transform(train['department'])
test['department'] = le.fit_transform(test['department'])
train['gender'] = le.fit_transform(train['gender'])
test['gender'] = le.fit_transform(test['gender'])


"""
  Splitting the Data
  We store the Target Variable in y, and then we store the rest of the columns in x,
  by deleting the target column from the data.
"""
y = train['is_promoted']
x = train.drop(['is_promoted'], axis=1)
x_test = test

# Resampling

# The Target class is Highly imbalanced.
# We use Over Sampling Technique to resample the data.
oversample = SMOTE()
x_resample, y_resample = oversample.fit_resample(x, y.values.ravel())


# split the test and train dataset into train and validity sets
x_train, x_valid, y_train, y_valid = train_test_split(x_resample, y_resample, test_size=0.2, random_state=0)


# Feature Scaling
# scale all the features of the dataset into the same scale using standard scaling
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_valid = sc.transform(x_valid)
x_test = sc.transform(x_test)


# use Decision Trees to classify the data
model = DecisionTreeClassifier()
model.fit(x_train, y_train)

y_pred = model.predict(x_valid)

# accuracy
# print("Accuracy: ", model.score(x_valid, y_valid))

# Dump(save) classifier and scalar to disk using pickle library
pickle.dump(model, open('Assets\\pickle\\model.pkl', 'wb'))
pickle.dump(sc, open('Assets\\pickle\\scalar.pkl', 'wb'))
