# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 22:40:09 2018

@author: bdebaque
"""



# %% Single line bayesian optimization of polynomial function
import numpy as np
from hyperopt import hp, tpe, fmin
# Hyperopt uses the Tree Parzen Estimator (TPE)

best = fmin(fn = lambda x: np.poly1d([1, -2, -28, 28, 12, -26, 100])(x),
            space = hp.normal('x', 4.9, 0.5), algo=tpe.suggest, 
            max_evals = 2000)

# %% Gradient Boosting Machine Hyperparameter Optimization
import pandas as pd
import numpy as np

# Modeling
import lightgbm as lgb

# Evaluation of the model
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

MAX_EVALS = 2
N_FOLDS = 2

# Read in data and separate into training and testing sets
features = pd.read_csv('../input/application_train.csv')

# Extract the labels and format properly
labels = np.array(features['TARGET'].astype(np.int32)).reshape((-1,))

# Drop the unneeded columns
features = features.drop(columns = ['SK_ID_CURR', 'TARGET'])

train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.2)

print('Train shape: ', train_features.shape)
print('Test shape: ', test_features.shape)

train_features.head()