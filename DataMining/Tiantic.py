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
print('{} 没有存活在 {} 游客中 并且占训练集的 {:.2f}% .'.format(not_survived, df_train.shape[0], not_survived_per))

plt.figure(figsize=(10, 8))
sns.countplot(df_train['Survived'])

plt.xlabel('Survival', size=15, labelpad=15)
plt.ylabel('Passenger Count', size=15, labelpad=15)
plt.xticks((0, 1), ['Not Survived ({0:.2f}%)'.format(not_survived_per), 'Survived ({0:.2f}%)'.format(survived_per)])
plt.tick_params(axis='x', labelsize=13)
plt.tick_params(axis='y', labelsize=13)

plt.title('训练集存活分布', size=15, y=1.05)

plt.show()

# 某些特征与目标的相关性

print(df_train[['Pclass', 'Survived']].groupby('Pclass', as_index=False)['Survived'].mean())
print()
print(df_train[['Sex', 'Survived']].groupby('Sex', as_index=False)['Survived'].mean())
print()

# 使用目标变量绘制连续特征图

cont_features = ['Age', 'Fare']
surv = df_train['Survived'] == 1

fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(30, 30))
plt.subplots_adjust(right=0.9)

for i, feature in enumerate(cont_features):
    # 生存分布特征
    sns.distplot(df_train[~surv][feature], label='Not Survived', hist=True, color='#e74c3c', ax=axs[0][i])
    sns.distplot(df_train[surv][feature], label='Survived', hist=True, color='#2ecc71', ax=axs[0][i])

    # 特征在数据集中的分布
    sns.distplot(df_train[feature], label='Training Set', hist=False, color='#e74c3c', ax=axs[1][i])
    #     sns.distplot(df_test[feature], label='Test Set', hist=False, color='#2ecc71', ax=axs[1][i])

    axs[0][i].set_xlabel('')
    axs[1][i].set_xlabel('')

    for j in range(2):
        axs[i][j].tick_params(axis='x', labelsize=20)
        axs[i][j].tick_params(axis='y', labelsize=20)

    axs[0][i].legend(loc='upper right', prop={'size': 20})
    axs[1][i].legend(loc='upper right', prop={'size': 20})
    axs[0][i].set_title('Distribution of Survival in {}'.format(feature), size=20, y=1.05)

axs[1][0].set_title('Distribution of {} Feature'.format('Age'), size=20, y=1.05)
axs[1][1].set_title('Distribution of {} Feature'.format('Fare'), size=20, y=1.05)

plt.show()

# 使用目标变量绘制分类特征图
cat_features = ['Embarked', 'Parch', 'Pclass', 'Sex', 'SibSp']

fig, axs = plt.subplots(ncols=2, nrows=3, figsize=(20, 20))
plt.subplots_adjust(right=0.9, top=0.8)

for i, feature in enumerate(cat_features, 1):
    plt.subplot(2, 3, i)
    sns.countplot(x=feature, hue='Survived', data=df_train)

    plt.xlabel('{}'.format(feature), size=20, labelpad=15)
    plt.ylabel('Passenger Count', size=20, labelpad=15)
    plt.tick_params(axis='x', labelsize=20)
    plt.tick_params(axis='y', labelsize=20)

    plt.legend(['Not Survived', 'Survived'], loc='upper center', prop={'size': 18})
    plt.title('Count of Survival in {} Feature'.format(feature), size=20, y=1.05)

plt.show()

"""大多数特征是相互关联的。此关系可用于创建具有特征转换和特征交互的新特征。目标编码也非常有用，因为它与幸存的特征有很高的相关性。
分割点和尖峰在连续特征中可见。决策树模型可以很容易地捕捉到它们，但线性模型可能无法发现它们。
分类特征具有非常明显的分布，具有不同的生存率。这些特征可以是一个热编码的。这些特性中的一些可以相互组合以形成新特性。"""

# 特征工程
df_all = concat_df(df_train, df_test)
df_all.head()

# 创建一些新功能
df_all['FamilySize'] = df_all.apply(lambda x: x['SibSp'] + x['Parch'] + 1, axis='columns')
print(df_all[['FamilySize', 'Survived']].groupby('FamilySize', as_index=False)['Survived'].mean())

print()

df_all['IsAlone'] = df_all.apply(lambda x: 1 if x['FamilySize'] == 1 else 0, axis='columns')
print(df_all[['IsAlone', 'Survived']].groupby('IsAlone', as_index=False)['Survived'].mean())

print()

# 票价
# #票价特征倾斜，右端存活率极高。
# #分成分位数箱
df_all['Fare'] = pd.qcut(df_all['Fare'], 13)


fig, axs = plt.subplots(figsize=(22, 9))
sns.countplot(x='Fare', hue='Survived', data=df_all)

plt.xlabel('Fare', size=15, labelpad=20)
plt.ylabel('Passenger Count', size=15, labelpad=20)
plt.tick_params(axis='x', labelsize=10)
plt.tick_params(axis='y', labelsize=15)

plt.legend(['Not Survived', 'Survived'], loc='upper right', prop={'size': 15})
plt.title('Count of Survival in {} Feature'.format('Fare'), size=15, y=1.05)

plt.show()

# Age
df_all['Age'] = pd.qcut(df_all['Age'], 10)

fig, axs = plt.subplots(figsize=(22, 9))
sns.countplot(x='Age', hue='Survived', data=df_all)

plt.xlabel('Age', size=15, labelpad=20)
plt.ylabel('Passenger Count', size=15, labelpad=20)
plt.tick_params(axis='x', labelsize=15)
plt.tick_params(axis='y', labelsize=15)

plt.legend(['Not Survived', 'Survived'], loc='upper right', prop={'size': 15})
plt.title('Survival Counts in {} Feature'.format('Age'), size=15, y=1.05)

plt.show()

# 我们不能直接使用票证功能，因为它很大，但我们可以使用num个人共享相同的票证号码
# #作为计数功能，作为参与方大小的代理
df_all['Ticket_Frequency'] = df_all.groupby('Ticket')['Ticket'].transform('count')

# Title and isMarried

df_all['Title'] = df_all['Name'].str.split(', ', expand=True)[1].str.split('.', expand=True)[0]
df_all['Title'] = df_all['Title'].replace(['Miss', 'Mrs','Ms', 'Mlle', 'Lady', 'Mme', 'the Countess', 'Dona'], 'Miss/Mrs/Ms')
df_all['Title'] = df_all['Title'].replace(['Dr', 'Col', 'Major', 'Jonkheer', 'Capt', 'Sir', 'Don', 'Rev'], 'Dr/Military/Noble/Clergy')

sns.barplot(x=df_all['Title'].value_counts().index, y=df_all['Title'].value_counts().values)
plt.show()


df_all['Is_Married'] = 0
df_all['Is_Married'].loc[df_all['Title'] == 'Mrs'] = 1

df_train, df_test = divide_df(df_all)

df_all.head()

# 标签编码非数字顺序特征
non_numeric_features = ['Age', 'Fare']

# for feature in non_numeric_features:
encoder = OrdinalEncoder()
df_train[non_numeric_features] = encoder.fit_transform(df_train[non_numeric_features])
df_test[non_numeric_features] = encoder.transform(df_test[non_numeric_features])

# 一个热编码分类功能
cat_features = ['Pclass', 'Sex', 'Embarked', 'Title']


for feature in cat_features:
    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    encoded_feat_train = pd.DataFrame(encoder.fit_transform(df_train[[feature]]))
    encoded_feat_test = pd.DataFrame(encoder.transform(df_test[[feature]]))

    # get OHE column names
    encoded_feat_train.columns = encoder.get_feature_names([feature])
    encoded_feat_test.columns = encoder.get_feature_names([feature])

    # One-hot encoding removed index; put it back
    encoded_feat_train.index = df_train.index
    encoded_feat_test.index = df_test.index

    # Remove categorical columns (will replace with one-hot encoding)
    rem_train = df_train.drop(feature, axis=1)
    rem_test = df_test.drop(feature, axis=1)

    # Add one-hot encoded columns to numerical features
    df_train = pd.concat([rem_train, encoded_feat_train], axis=1)
    df_test = pd.concat([rem_test, encoded_feat_test], axis=1)

# 丢弃不需要的列
drop_cols = ['Name', 'Ticket', 'SibSp', 'Parch']
df_train.drop(drop_cols, axis=1, inplace=True)
df_test.drop(drop_cols, axis=1, inplace=True)


df_train.shape, df_test.shape

print(sorted(df_train.columns))
print()

print(sorted(df_test.columns))

df_train



# 模型
df_val = pd.read_csv("./titanic/gender_submission.csv")
train_X = df_train.drop(['Survived', 'PassengerId'], axis=1)
train_y = df_train['Survived']
test_X = df_test.drop(['PassengerId'], axis=1).copy()
test_y = df_val['Survived']


print('X_train shape: {}'.format(train_X.shape))
print('y_train shape: {}'.format(train_y.shape))
print('X_test shape: {}'.format(test_X.shape))


# Logistic Regression
logit = LogisticRegression()
logit.fit(train_X, train_y)
pred_y = logit.predict(test_X)

# training loss
acc_log = round(logit.score(train_X, train_y) * 100, 2)
print(acc_log)

# logistic regression weights per feature signify a positive or negative effect toward survival
    # Title_Master(kid), Sex_Female, 1st class have high positive weights toward survival
    # Large families, Sex_Male, 3rd class have high negative weights toward survival

coeff_df = pd.DataFrame(train_X.columns)
coeff_df.columns = ['Feature']
coeff_df["Weights"] = pd.Series(logit.coef_[0])
coeff_df.sort_values(by='Weights', ascending=False)

# 决策树
decision_tree = DecisionTreeClassifier()
decision_tree.fit(train_X, train_y)
tree_pred_y = decision_tree.predict(test_X)
acc_decision_tree = round(decision_tree.score(train_X, train_y) * 100, 2)
acc_decision_tree_test = round(decision_tree.score(test_X, test_y)*100, 2)
print("---------------decision_tree----------------")
print("训练准确度:{}".format(acc_decision_tree))
print("测试集验证准确度:{}".format(acc_decision_tree_test))
print("---------------decision_tree----------------")


# 随机森林
random_forest = RandomForestClassifier(
    criterion='gini',
    n_estimators=500,
    max_depth=10,
    min_samples_split=4,
    min_samples_leaf=5,
    max_features='auto',
    oob_score=True,
    random_state=SEED,
    n_jobs=-1,
    verbose=1
)
random_forest.fit(train_X, train_y)
rf_pred_y = random_forest.predict(test_X)
acc_random_forest = round(random_forest.score(train_X, train_y) * 100, 2)
acc_random_forest_test = round(random_forest.score(test_X, test_y)*100, 2)
print("---------------random_forest----------------")
print("训练准确度:{}".format(acc_random_forest))
print("测试集验证准确度:{}".format(acc_random_forest_test))
print("---------------random_forest----------------")

submission_tree = pd.DataFrame({
    "PassengerId": df_test["PassengerId"],
    "Survived": tree_pred_y.astype(int)
})
submission_tree.head(10)
submission_tree.to_csv('submission_tree.csv', header=True, index=False)
x = pd.read_csv('submission_tree.csv')
x['Survived'].value_counts()

submission_random = pd.DataFrame({
    "PassengerId": df_test["PassengerId"],
    "Survived": rf_pred_y.astype(int)
})
submission_random.head(10)
submission_random.to_csv('submission_random.csv', header=True, index=False)
x = pd.read_csv('submission_random.csv')
x['Survived'].value_counts()