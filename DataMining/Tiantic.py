import numpy as np
import pandas as pd

import os
for dirname, _, filenames in os.walk('titanic'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

import string
import warnings
warnings.filterwarnings('ignore')

SEED = 42

def concat_df(train_data, test_data):
    # Returns a concatenated df of training and test set
    return pd.concat([train_data, test_data], sort=True).reset_index(drop=True)

def divide_df(all_data):
    # Returns divided dfs of training and test set
    return all_data.loc[:890], all_data.loc[891:].drop(['Survived'], axis=1)

# Load in the train and test datasets
df_train = pd.read_csv('titanic/train.csv')
df_test = pd.read_csv('titanic/test.csv')

# will be used for analysis
df_all = concat_df(df_train, df_test)

df_train.name = 'Training Set'
df_test.name = 'Test Set'
df_all.name = 'All Set'

dfs = [df_train, df_test]

print('训练集例子数 = {}'.format(df_train.shape[0]))
print('测试集例子数 = {}\n'.format(df_test.shape[0]))
print('Training X Shape = {}'.format(df_train.shape))
print('Training y Shape = {}\n'.format(df_train['Survived'].shape[0]))
print('Test X Shape = {}'.format(df_test.shape))
print('Test y Shape = {}\n'.format(df_test.shape[0]))
print(df_train.columns)
print(df_test.columns)

df_train.head()

df_train.info()
df_test.info()

# 缺失值
def display_missing(df):
    for col in df.columns:
        print('{} column missing values: {}'.format(col, df[col].isnull().sum()))
    print('\n')


for df in [df_train, df_test]:
    print('{}'.format(df.name))
    display_missing(df)

    # Age
    # Age中缺失的值用中位数年龄来填充，但是使用整个数据集的中位数年龄并不是一个好的选择
    # Pclass组中位年龄与年龄相关性高，为最佳选择(0.408106)

    # 所有特征与所有其他特征的相关性
    df_train_corr = df_train.corr().abs().unstack().sort_values(kind="quicksort", ascending=False).reset_index()
    df_train_corr.rename(columns={"level_0": "Feature 1", "level_1": "Feature 2", 0: 'Correlation Coefficient'},
                         inplace=True)
    print(df_train_corr[df_train_corr['Feature 1'] == 'Age'])

    # 训练台和测试台分开使用，避免泄漏
    df_train['Age'] = df_train.groupby(['Sex', 'Pclass'])['Age'].apply(lambda x: x.fillna(x.median()))
    df_test['Age'] = df_test.groupby(['Sex', 'Pclass'])['Age'].apply(lambda x: x.fillna(x.median()))

# Embarked 登船港口
# 寻找缺失值并用最频繁的填充
print(df_all[df_all['Embarked'].isnull()])
df_train['Embarked'] = df_train['Embarked'].fillna('S')

# Fare
print(df_all[df_all['Fare'].isnull()])
med_fare = df_test.groupby(['Pclass', 'Parch', 'SibSp']).Fare.median()[3][0][0]
# “票价”中所缺的数值，以三等乘客票价的中位数填补
df_test['Fare'] = df_test['Fare'].fillna(med_fare)

# Cabin
# 太多缺失值，丢弃
df_train.drop(['Cabin'], axis=1, inplace=True)
df_test.drop(['Cabin'], axis=1, inplace=True)
df_train.name = 'Training Set'
df_test.name = 'Test Set'

# 再次验证是否还有缺失值
def display_missing(df):
    for col in df.columns:
        print('{} column missing values: {}'.format(col, df[col].isnull().sum()))
    print('\n')


for df in [df_train, df_test]:
    print('{}'.format(df.name))
    display_missing(df)


# 目标和特征分布
survived = df_train['Survived'].value_counts()[1]
not_survived = df_train['Survived'].value_counts()[0]
survived_per = survived / df_train.shape[0] * 100
not_survived_per = not_survived / df_train.shape[0] * 100

print('{} 存活在 {}游客中  并且占训练集的 {:.2f}% .'.format(survived, df_train.shape[0], survived_per))
print('{} 存活在 {} 游客中 并且占训练集的 {:.2f}% .'.format(not_survived, df_train.shape[0], not_survived_per))

plt.figure(figsize=(10, 8))
sns.countplot(df_train['Survived'])

plt.xlabel('Survival', size=15, labelpad=15)
plt.ylabel('Passenger Count', size=15, labelpad=15)
plt.xticks((0, 1), ['Not Survived ({0:.2f}%)'.format(not_survived_per), 'Survived ({0:.2f}%)'.format(survived_per)])
plt.tick_params(axis='x', labelsize=13)
plt.tick_params(axis='y', labelsize=13)

plt.title('训练集存活分布', size=15, y=1.05)

plt.show()

# 与其他标签之间的关联

print(df_train[['Pclass', 'Survived']].groupby('Pclass', as_index=False)['Survived'].mean())
print()
print(df_train[['Sex', 'Survived']].groupby('Sex', as_index=False)['Survived'].mean())
print()