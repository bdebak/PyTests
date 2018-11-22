# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 23:58:33 2018

@author: bdebaque
"""

import pandas as pd
import numpy as np

df = pd.read_csv('D:/Datasets/Kaggle/BigML/telecom_churn.csv')
df.head()

print(df.shape)
print(df.columns)
print(df.info())

# change column type
df['churn'] = df['churn'].astype('int64')

df.describe()

# stats on non numerical features
df.describe(include=['object', 'bool'])

df['churn'].value_counts()

df['churn'].value_counts(normalize=True)

df.sort_values(by='total day charge', ascending=False).head()

df.sort_values(by=['churn', 'total day charge'], ascending=[True, False]).head()

df['churn'].mean()

df[df['churn'] == 1].mean()

df[df['churn'] == 1]['total day minutes'].mean()

df[(df['churn'] == 0) & (df['international plan'] == 'no')]['total intl minutes'].max()

df.loc[0:5, 'state':'area code']

df.iloc[0:5, 0:3]

df.apply(np.max)

df[df['state'].apply(lambda state: state[0] == 'W')].head()

d = {'no' : False, 'yes' : True} 
df['international plan'] = df['international plan'].map(d) 
df.head()

df = df.replace({'voice mail plan': d}) 
df.head()

df.groupby(by=grouping_columns)[columns_to_show].function()

columns_to_show = ['total day minutes', 'total eve minutes', 
                   'total night minutes']
df.groupby(['churn'])[columns_to_show].describe(percentiles=[])

columns_to_show = ['total day minutes', 'total eve minutes', 
                   'total night minutes']
df.groupby(['churn'])[columns_to_show].agg([np.mean, np.std, 
                                            np.min, np.max])

pd.crosstab(df['churn'], df['international plan'])

pd.crosstab(df['churn'], df['voice mail plan'], normalize=True)

df.pivot_table(['total day calls', 'total eve calls', 'total night calls'], ['area code'], aggfunc='mean')

total_calls = df['total day calls'] + df['total eve calls'] + df['total night calls'] + df['total intl calls'] 
df.insert(loc=len(df.columns), column='total calls', value=total_calls) 
df.head()

df['total charge'] = df['total day charge'] + df['total eve charge'] + df['total night charge'] + df['total intl charge']
df.head()

# get rid of just created columns 
df.drop(['total charge', 'total calls'], axis=1, inplace=True) 
# and here’s how you can delete rows 
df.drop([1, 2]).head()

# %% Prediction
pd.crosstab(df['churn'], df['international plan'], margins=True)
# some imports and "magic" commands to set up plotting 
import matplotlib.pyplot as plt 
# pip install seaborn 
import seaborn as sns

plt.rcParams['figure.figsize'] = (8, 6)
sns.countplot(x='international plan', hue='churn', data=df)
pd.crosstab(df['churn'], df['customer service calls'], margins=True)

df['Many_service_calls'] = (df['customer service calls'] > 3).astype('int')
pd.crosstab(df['Many_service_calls'], df['churn'], margins=True)
sns.countplot(x='Many_service_calls', hue='churn', data=df);

pd.crosstab(df['churn'], df['customer service calls'], margins=True)


