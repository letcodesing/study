# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# %matplotlib inline
import matplotlib.pyplot as plt  # Matlab-style plotting
import seaborn as sns
color = sns.color_palette()
sns.set_style('darkgrid')
import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)

from scipy import stats
from scipy.stats import norm, skew #for some statistics

pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x)) #Limiting floats output to 3 decimal points

train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
submit = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv')
print(train.shape, test.shape, submit.shape)
display(train.head(), test.head())

train = train.drop(['Id'], axis=1)
test = test.drop(['Id'], axis=1)
train.shape, test.shape

fig, ax = plt.subplots()
ax.scatter(x = train['GrLivArea'], y = train['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)
plt.show()

# 이상치 제거하기
train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)

fig, ax = plt.subplots()
ax.scatter(x = train['GrLivArea'], y = train['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)
plt.show()

sns.distplot(train['SalePrice'] , fit=norm);

# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(train['SalePrice'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

#Now plot the distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')

#Get also the QQ-plot
fig = plt.figure()
res = stats.probplot(train['SalePrice'], plot=plt)
plt.show()

train['SalePrice'] = np.log1p(train['SalePrice'])

sns.distplot(train['SalePrice'] , fit=norm);

# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(train['SalePrice'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

#Now plot the distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')

#Get also the QQ-plot
fig = plt.figure()
res = stats.probplot(train['SalePrice'], plot=plt)
plt.show()

label = train['SalePrice']
all_data = pd.concat((train,test)).reset_index(drop=True)
all_data = all_data.drop(['SalePrice'],axis=1)
print(all_data.shape)
all_data

# 결측값 조회하기
(all_data.isnull().sum() / len(all_data) * 100).sort_values(ascending=False)[ :25]

all_data_na = (all_data.isnull().sum() / len(all_data) * 100).sort_values(ascending=False)[:25]
all_data_na

f, ax = plt.subplots(figsize=(15, 5))
plt.xticks(rotation='90')
sns.barplot(x=all_data_na.index, y=all_data_na)
plt.xlabel('Features', fontsize=15)
plt.ylabel('Percent of missing values', fontsize=15)
plt.title('Percent missing data by feature', fontsize=15)

#Correlation map to see how features are correlated with SalePrice
corrmat = train.corr()
plt.subplots(figsize=(12,9))
sns.heatmap(corrmat, vmax=0.9, square=True)

all_data['MiscFeature']

all_data['PoolQC'].value_counts() # 각 성분별 개수 확인
# all_data['PoolQC'].unique() # 각 성분의 종류 확인
# all_data['PoolQC'].nunique() # 각 성분의 종류의 개수 확인

all_data['PoolQC'] = all_data['PoolQC'].fillna('None')
all_data['MiscFeature'] = all_data['MiscFeature'].fillna('None')
all_data['LotFrontage'].isnull().sum()

all_data['LotFrontage'].value_counts()
all_data['LotFrontage'] = all_data['LotFrontage'].fillna(all_data['LotFrontage'].median())
all_data['KitchenQual'].value_counts()

for col in ['MSZoning', 'KitchenQual', 'Exterior1st', 'Exterior2nd', 'SaleType']:
    all_data[col] = all_data[col].fillna(all_data[col].mode()[0])
all_data

all_data.info()

cat_col = all_data.dtypes[all_data.dtypes == 'object'].index
for col in cat_col:
    all_data[col] = all_data[col].fillna('None')
# # data의 전체적인 분포 살펴보기 (sweetviz)
# # !pip install sweetviz
# # import sweetviz as sv

# report = sv.analyze(all_data)
# report.show_html(filepath='sv_output_analyze.html')
# # report.show_notebook()
all_data.dtypes[(all_data.dtypes == 'int64') | (all_data.dtypes == 'float64')].index

num_col = list(all_data.dtypes[(all_data.dtypes == 'int64') | (all_data.dtypes == 'float64')].index) # 숫자로 된 열이름 추출
for col in num_col:
    all_data[col] = all_data[col].fillna(0)
(all_data.isnull().sum() / len(all_data) * 100).sort_values(ascending=False)[:5] # 결측치가 모두 채워진 것을 확인

all_data.head()

cat_col[0]


all_data['MSZoning'].values


# 1-6 Label encoding (문자를 전부 숫자로 변경)
from sklearn.preprocessing import LabelEncoder
lbl = LabelEncoder()
cat_col = list(all_data.dtypes[all_data.dtypes == 'object'].index) # 문자열로 된 열이름 추출
for col in cat_col:
    all_data[col] = lbl.fit_transform(all_data[col].values)
all_data.head(2)

all_data['MSSubClass'] = all_data['MSSubClass'].astype('category')
# 숫자이지만 건축물의 종류를 구분하는 번호이므로 문자로 변경
all_data['MSSubClass'] = all_data['MSSubClass'].astype('category')
# 월은 대소비교보단 계절성을 나타내기 위해 숫자를 문자로 변경 (범주화)
all_data['MoSold'] = all_data['MoSold'].astype('category')
# all_data['MoSold'].value_counts()
all_data.dtypes

# Adding total square feet feature (집의 전체 넓이)
all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']
# Skewed features 치우진 분포의 열을 정규분포에 가까운 모양으로 변환하기
num_col = list(all_data.dtypes[(all_data.dtypes == 'int64') | (all_data.dtypes == 'float64')].index)
num_col

all_data[num_col].head(3)

all_data[num_col].apply(lambda x : skew(x)).sort_values(ascending=False)


skewed_feat = all_data[num_col].apply(lambda x : skew(x)).sort_values(ascending=False)
skewed_feat, len(skewed_feat)

skewed_feat = skewed_feat[abs(skewed_feat) > 0.75]
print(len(skewed_feat))
skewed_feat.index

skewed_feat = skewed_feat[abs(skewed_feat) > 0.75]
print(len(skewed_feat))
from scipy.special import boxcox1p
skewed_col = skewed_feat.index
for col in skewed_col:
    all_data[col] = boxcox1p(all_data[col], 0.15) # 0.15는 lambda 값으로 변형 정도를 결정
    
    # Model libraries import
from sklearn.linear_model import ElasticNet, Lasso, BayesianRidge, LassoLarsIC, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb
all_data

# indexing, slicing 하기
a = [1,2,3,4,5,6]
a[-1]

# 학습을 위해 all_data를 train과 test로 다시 분할
train = all_data[:len(train)]
test = all_data[len(train):]
print(train.shape, test.shape)

# 교차검증시 train, test를 split 하지 않음
X = train
y = label # train['SalePrice']
# train_test splitting
from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state=0, test_size=0.2)
# split 결과 확인
X_train.shape, X_valid.shape, y_train.shape, y_valid.shape

# model = Ridge(random_state = 0) # 20967 / 21806
model = Lasso(alpha=0.0005, random_state=1) # 21192 / 21667 / 0.13039
# model = ElasticNet(alpha=0.0005, l1_ratio=0.9, random_state=0) # 21152 / 21676
# model = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5) # 9582 / 38248
# model = RandomForestRegressor(random_state=0) # 10038 / 23224 / 0.14737
# model = lgb.LGBMRegressor(objective='regression', num_leaves=5, learning_rate=0.05, # 12626 / 20465 / 0.12816
#                         n_estimators=720, max_bin=55, bagging_fraction=0.8,
#                         bagging_freq=5, feature_fraction=0.23, feature_fraction_seed=9,
#                         bagging_seed=9, min_data_in_leaf=6, min_sum_hessian_in_leaf=11)
# RandomForest, LGBM의 1:1 Ensenble 모델의 test score : 0.13288
# Lasso, LGBM의 1:1 Ensenble 모델의 test score : 0.12471
# from sklearn.svm import SVR # regression
# model = SVR(C=100, gamma='auto', degree=3, epsilon=.1,coef0=1)
# model = SVR() # 48663 / 50628

model.fit(X_train, y_train) # 모델 학습
pred_train = np.expm1(model.predict(X_train)) # 훈련 부분 예측
pred_valid = np.expm1(model.predict(X_valid)) # 검증 부분 예측
pred_test = np.expm1(model.predict(test)) # 실제 부분 예측
# rmse score 조회 (log1p 변환 됐던 걸 expm1 으로 복구해줘서 rmsle가 아닌 rmse 로 측정)
train_score = mean_squared_error(np.expm1(y_train), pred_train) ** 0.5
valid_score = mean_squared_error(np.expm1(y_valid), pred_valid) ** 0.5
print('train score / valid score')
print(int(train_score),'/', int(valid_score))

from sklearn.linear_model import ElasticNet, Lasso, BayesianRidge, LassoLarsIC, Ridge
# RandomForest, LGBM의 1:1 Ensenble 모델 만들기
Lasso = Lasso(alpha=0.0005, random_state=1)
Lasso.fit(X_train, y_train) # 모델 학습
pred_Lasso = np.expm1(Lasso.predict(test)) # 실제 부분 예측
#--------------------------------------------
LGBM = lgb.LGBMRegressor(objective='regression', num_leaves=5, learning_rate=0.05,
                        n_estimators=720, max_bin=55, bagging_fraction=0.8,
                        bagging_freq=5, feature_fraction=0.23, feature_fraction_seed=9,
                        bagging_seed=9, min_data_in_leaf=6, min_sum_hessian_in_leaf=11)
LGBM.fit(X_train, y_train) # 모델 학습
pred_LGBM = np.expm1(LGBM.predict(test)) # 실제 부분 예측
#--------------------------------------------
pred_Lasso_LGBM = (pred_Lasso + pred_LGBM) / 2
print(pred_Lasso[:3])
print(pred_LGBM[:3])
print(pred_Lasso_LGBM[:3])

submit = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv')
submit.head(3)

submit['SalePrice'] = pred_Lasso_LGBM
submit.to_csv('submit_Lasso_LGBM.csv', index=False)
submit.head()

# !pip install pycaret
# Dataset = X.copy()
# Dataset['SalePrice'] = label
# Dataset
# from pycaret.regression import *
# import pycaret
# setup_reg = setup(data=Dataset, target='SalePrice')
# pip list

# https://www.kaggle.com/code/laplace8/house-price-practice-220628