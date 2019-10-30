
# https://www.kaggle.com/ldfreeman3/a-data-science-framework-to-achieve-99-accuracy


# Define the Problem:
# Gather the Data:
# Prepare Data for Consumption:
# The 4 C's of Data Cleaning: Correcting, Completing, Creating, and Converting
# Perform Exploratory Analysis:
# Model Data:
# Validate and Implement Data Model:
# Optimize and Strategize:

# Import packages
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

import IPython
from IPython import display  # pretty printing of dataframes in Jupyter notebook

import sklearn
import random
import time

# Import libraries
import random
import time

# Common Model Algorithms
from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process
from xgboost import XGBClassifier

# Common Model Helpers
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn import feature_selection
from sklearn import model_selection
from sklearn import metrics

# Visualization
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns
from pandas.plotting import scatter_matrix


mpl.style.use('ggplot')
sns.set_style('white')
pylab.rcParams['figure.figsize'] = 12, 8

# ignore warnings
import warnings
warnings.filterwarnings('ignore')
print('-' * 25)

# LOOKING AT THE DATA ------------------------------------------------------------------------------

data_dir = "C:/Users/Khoa/Desktop/Coding/0.1-Project-Data/Baseline_1.0/Titanic/"

data_raw = pd.read_csv(data_dir + 'train.csv')
data_val = pd.read_csv(data_dir + 'test.csv')

data1 = data_raw.copy(deep=True)

data_cleaner = [data1, data_val]

pd.options.display.width = 0

print(data_raw.info())
print(data_raw.head(10))
print(data_raw.sample(10))
print('Train columns with null values:\n', data1.isnull().sum())
print("-" * 10)
print('Test/Validation columns with null values:\n', data_val.isnull().sum())
print("-" * 10)
print(data_raw.describe(include='all'))

# LOOKING AT THE DATA ------------------------------------------------------------------------------

for dataset in data_cleaner:
    dataset['Age'].fillna(dataset['Age'].median(), inplace=True)
    dataset['Embarked'].fillna(dataset["Embarked"].mode()[0], inplace=True)
    dataset['Fare'].fillna(dataset['Fare'].median(), inplace=True)

drop_column = ["PassengerId", "Cabin", "Ticket"]
data1.drop(drop_column, axis=1, inplace=True)

print(data1.head(10))

print(data1.isnull().sum())
print("-" * 10)

# FEATURE ENGINEERING ------------------------------------------------------------------------------

for dataset in data_cleaner:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
    dataset['IsAlone'] = 1
    dataset['IsAlone'].loc[dataset['FamilySize'] > 1] = 0
    dataset['Title'] = dataset['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
    dataset['AgeBin'] = pd.cut(dataset['Age'].astype(int), 5)
    dataset['FareBin'] = pd.qcut(dataset['Fare'], 4)

print(dataset.head(10))

stat_min = 10
title_names = (data1['Title'].value_counts() < stat_min)

data1['Title'] = data1['Title'].apply(lambda x: 'Misc' if title_names.loc[x] == True else x)
print(data1['Title'].value_counts())
print("-" * 10)

# data1.info()
# data_val.info()
# data1.sample(10)

# DEFINE x (independent/features/explanatory/predictor/etc.) and y (dependent/target/outcome/response/etc.)   ------------------------------------------------------------------------------

label = LabelEncoder()
for dataset in data_cleaner:
    dataset['Sex_Code'] = label.fit_transform(dataset['Sex'])
    dataset['Embarked_Code'] = label.fit_transform(dataset['Embarked'])
    dataset['Title_Code'] = label.fit_transform(dataset['Title'])
    dataset['AgeBin_Code'] = label.fit_transform(dataset['AgeBin'])
    dataset['FareBin_Code'] = label.fit_transform(dataset['FareBin'])

print(dataset.head(10))

Target = ['Survived']

data1_x = ['Sex', 'Pclass', 'Embarked', 'Title', 'SibSp', 'Parch', 'Age', 'Fare', 'FamilySize', 'IsAlone']  # pretty name/values for charts
data1_x_calc = ['Sex_Code', 'Pclass', 'Embarked_Code', 'Title_Code', 'SibSp', 'Parch', 'Age', 'Fare']  # coded for algorithm calculation
data1_xy = Target + data1_x
print('Original X Y: ', data1_xy, '\n')

# define x variables for original w/bin features to remove continuous variables
data1_x_bin = ['Sex_Code', 'Pclass', 'Embarked_Code', 'Title_Code', 'FamilySize', 'AgeBin_Code', 'FareBin_Code']
data1_xy_bin = Target + data1_x_bin
print('Bin X Y: ', data1_xy_bin, '\n')

# define x and y variables for dummy features original
data1_dummy = pd.get_dummies(data1[data1_x])
data1_x_dummy = data1_dummy.columns.tolist()
data1_xy_dummy = Target + data1_x_dummy

print('Dummy X Y: ', data1_xy_dummy, '\n')
print(data1_dummy.head())

print('Train columns with null values: \n', data1.isnull().sum())
print("-" * 10)
print(data1.info())
print("-" * 10)

print('Test/Validation columns with null values: \n', data_val.isnull().sum())
print("-" * 10)
print(data_val.info())
print("-" * 10)

data_raw.describe(include='all')


#  Split Training and Testing Data ------------------------------------------------------------------------------
