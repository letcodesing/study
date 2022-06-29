# https://log-laboratory.tistory.com/194

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm_notebook
import matplotlib as mpl
from scipy import stats
from sklearn.preprocessing import LabelEncoder

# 노트북 안에 그래프를 그리기 위해

# 그래프에서 격자로 숫자 범위가 눈에 잘 띄도록 ggplot 스타일을 사용
plt.style.use('ggplot')

# 그래프에서 마이너스 폰트 깨지는 문제에 대한 대처
mpl.rcParams['axes.unicode_minus'] = False

house_train = pd.read_csv('./_data/kaggle_house/train.csv', encoding='utf-8')
house_train.head()

house_train.shape
house_train.columns
#데이터 컬럼 형식 확인
house_train.info()
# 데이터 요약
house_train.describe() 
#null 확인
house_train.isnull().sum()
# NULL 확인
import missingno as msno
#모든 컬럼에 대한 결측치
msno.matrix(house_train, figsize=(12,5))

# 독립변수
# 수치형, 연속형 변수 구분
# data description 문서와 실제 train.csv 파일의 column이 다른것 변경 ('Kitchen'->'KitchenAbvGr', 'Bedroom'->'BedroomAbvGr')
# data type : serial 
numerical_col_df = house_train[['Id','SalePrice','LotFrontage','LotArea','OverallQual','OverallCond','MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','1stFlrSF','2ndFlrSF','LowQualFinSF','GrLivArea','BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr','KitchenAbvGr','TotRmsAbvGrd','Fireplaces','GarageCars','GarageArea','WoodDeckSF','OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea','MiscVal']]
category_col_df = house_train[['Id','MSSubClass','MSZoning','Street','Alley','LotShape','LandContour','Utilities','LotConfig','LandSlope','Neighborhood','Condition1','Condition2','BldgType','HouseStyle','YearBuilt','YearRemodAdd','RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','ExterQual','ExterCond','Foundation','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','Heating','HeatingQC','CentralAir','Electrical','KitchenQual','Functional','FireplaceQu','GarageType','GarageYrBlt','GarageFinish','GarageQual','GarageCond','PavedDrive','PoolQC','Fence','MiscFeature','MoSold','YrSold','SaleType','SaleCondition']]
category_date_col_df = house_train[['Id','YearBuilt','YearRemodAdd','GarageYrBlt','MoSold','YrSold']]

#데이터 형에 따른 결측치 확인 
#수치형 데이터 LotFrontage, MasVnrArea
msno.matrix(numerical_col_df, figsize=(12,5))

#범주형 데이터
#Alley, MasVnrType, Bsmt~관련, Fireplace 등등
msno.matrix(category_col_df, figsize=(20,5))

# column list , data type : list
numerical_col_list = ["SalePrice","LotFrontage","LotArea","OverallQual","OverallCond","MasVnrArea","BsmtFinSF1","BsmtFinSF2","BsmtUnfSF","TotalBsmtSF","1stFlrSF","2ndFlrSF","LowQualFinSF","GrLivArea","BsmtFullBath","BsmtHalfBath","FullBath","HalfBath","BedroomAbvGr","KitchenAbvGr","TotRmsAbvGrd","Fireplaces","GarageCars","GarageArea","WoodDeckSF","OpenPorchSF","EnclosedPorch","3SsnPorch","ScreenPorch","PoolArea","MiscVal"]
category_col_list = ["MSSubClass","MSZoning","Street","Alley","LotShape","LandContour","Utilities","LotConfig","LandSlope","Neighborhood","Condition1","Condition2","BldgType","HouseStyle","RoofStyle","RoofMatl","Exterior1st","Exterior2nd","MasVnrType","ExterQual","ExterCond","Foundation","BsmtQual","BsmtCond","BsmtExposure","BsmtFinType1","BsmtFinType2","Heating","HeatingQC","CentralAir","Electrical","KitchenQual","Functional","FireplaceQu","GarageType","GarageFinish","GarageQual","GarageCond","PavedDrive","PoolQC","Fence","MiscFeature","SaleType","SaleCondition"]
category_date_col_list = ["YearBuilt","YearRemodAdd","GarageYrBlt","MoSold","YrSold"]

category_col_df.dtypes

## 결측치 처리

## 수치형(지금은 0으로, 나중에 평균값으로 처리 하는방법도..)
numerical_col_df.LotFrontage.fillna('0', inplace=True)
## 범주형 (string 대체)
category_col_df.fillna('missing', inplace=True)

# 'MSSubClass' 숫자형으로된 범주형 제외
category_col_list = ["MSZoning","Street","Alley","LotShape","LandContour","Utilities","LotConfig","LandSlope","Neighborhood","Condition1","Condition2","BldgType","HouseStyle","RoofStyle","RoofMatl","Exterior1st","Exterior2nd","MasVnrType","ExterQual","ExterCond","Foundation","BsmtQual","BsmtCond","BsmtExposure","BsmtFinType1","BsmtFinType2","Heating","HeatingQC","CentralAir","Electrical","KitchenQual","Functional","FireplaceQu","GarageType","GarageFinish","GarageQual","GarageCond","PavedDrive","PoolQC","Fence","MiscFeature","SaleType","SaleCondition"]

category_col_c_df = category_col_df.copy()
# category_col_c_df.fillna('missing', inplace=True)

def cate_to_num(data, col_name,list):
    le = LabelEncoder()
    le.fit(list)
    data[col_name] = le.transform(data[col_name]) 
    return data

for category_col in category_col_list:
    # 결측치 어떻게 다룰지 고민.. 우선 string으로 처리.
    col_value_list = category_col_c_df[category_col].unique().tolist()    
    print("list : ",col_value_list)
    category_col_c_df = cate_to_num(category_col_c_df,category_col,col_value_list)    


print("data : ",category_col_c_df)

#16~17
##### 범주형 변수와 가격사이의 비교
figure, ((ax1,ax2,ax3,ax4), (ax5,ax6,ax7,ax8)) = plt.subplots(nrows=2, ncols=4)
figure.set_size_inches(18,8)

sns.barplot(data=house_train, x="MSSubClass", y="SalePrice", ax=ax1)
sns.barplot(data=house_train, x="LotShape", y="SalePrice", ax=ax2)
sns.barplot(data=house_train, x="BldgType", y="SalePrice", ax=ax3)
sns.barplot(data=house_train, x="HouseStyle", y="SalePrice", ax=ax4)
sns.barplot(data=house_train, x="KitchenAbvGr", y="SalePrice", ax=ax5)
sns.barplot(data=house_train, x="Functional", y="SalePrice", ax=ax6)
sns.barplot(data=house_train, x="SaleType", y="SalePrice", ax=ax7)
sns.barplot(data=house_train, x="SaleCondition", y="SalePrice", ax=ax8)

#ax1.set(ylabel='주거타입',title="연도별 가격")
# ax2.set(xlabel='부지모양',title="월별 가격")
# ax3.set(xlabel='주거 타입', title="일별 가격")
# ax4.set(xlabel='주거 스타일', title="시간별 가격")
# ax4.set(xlabel='최초공사년도', title="시간별 가격")
# ax4.set(xlabel='리모델링년도', title="시간별 가격")

# 시간에 따른 가격 분석
fig, axes = plt.subplots(nrows=2,ncols=2)
fig.set_size_inches(12, 10)
sns.boxplot(data=house_train,y="SalePrice",orient="v",ax=axes[0][0])
sns.boxplot(data=house_train,y="SalePrice",x="YearBuilt",orient="v",ax=axes[0][1])
sns.boxplot(data=house_train,y="SalePrice",x="MoSold",orient="v",ax=axes[1][0])
sns.boxplot(data=house_train,y="SalePrice",x="YrSold",orient="v",ax=axes[1][1])

# axes[0][0].set(ylabel='Count',title="대여량")
# axes[0][1].set(xlabel='Season', ylabel='Count',title="계절별 대여량")
# axes[1][0].set(xlabel='Hour Of The Day', ylabel='Count',title="시간별 대여량")
# axes[1][1].set(xlabel='Working Day', ylabel='Count',title="근무일 여부에 따른 대여량")

# 시간의 흐름에 따른 데이터 분석
house_train["YearBuilt"].value_counts()

fig,(ax1,ax2,ax3)= plt.subplots(nrows=3)
fig.set_size_inches(18,25)

sns.pointplot(data=house_train, x="MoSold", y="SalePrice", ax=ax1)

sns.pointplot(data=house_train, x="MoSold", y="SalePrice", hue="CentralAir", ax=ax2)

sns.pointplot(data=house_train, x="MoSold", y="SalePrice", hue="KitchenQual", ax=ax3)

# 면적과 가격의 상관관계 분석

corrMatt_area = house_train[["LotArea","MasVnrArea","BsmtFinSF1","BsmtFinSF2","BsmtUnfSF","TotalBsmtSF","1stFlrSF","2ndFlrSF","LowQualFinSF","GrLivArea","GarageArea","WoodDeckSF","OpenPorchSF","EnclosedPorch","3SsnPorch","ScreenPorch","PoolArea","SalePrice"]]
corrMatt_area = corrMatt_area.corr()
print(corrMatt_area)

mask = np.array(corrMatt_area)
mask[np.tril_indices_from(mask)] = False

# 면적과 가격의 상관관계 그래프
fig, ax = plt.subplots()
fig.set_size_inches(20,10)
sns.heatmap(corrMatt_area, mask=mask,vmax=.8, square=True,annot=True)

# 주택구성과 가격의 상관관계 그래프
# corrMatt_area = house_train[["Bedroom","Kitchen","TotRmsAbvGrd","Functional","Fireplaces","GarageCond","GarageType","GarageYrBlt","BldgType","HouseStyle","OverallQual","OverallCond","HeatingQC","Neighborhood","Condition1","Condition2","RoofMatl","SalePrice"]]
corrMatt_area = house_train[["TotRmsAbvGrd","Functional","Fireplaces","GarageCond","GarageType","GarageYrBlt","BldgType","HouseStyle","OverallQual","OverallCond","HeatingQC","Neighborhood","Condition1","Condition2","RoofMatl","SalePrice"]]
corrMatt_area = corrMatt_area.corr()
print(corrMatt_area)

mask = np.array(corrMatt_area)
mask[np.tril_indices_from(mask)] = False

# 주택구성과 가격의 상관관계 그래프
fig, ax = plt.subplots()
fig.set_size_inches(20,10)
sns.heatmap(corrMatt_area, mask=mask,vmax=.8, square=True,annot=True)


# trainWithoutOutliers
house_trainWithoutOutliers = house_train[np.abs(house_train["SalePrice"] - house_train["SalePrice"].mean()) <= (3*house_train["SalePrice"].std())]

print(house_train.shape)
print(house_trainWithoutOutliers.shape)

# saleprice 가격 분포도 파악(정규분포 적용, 중심 극한 정리)
figure, axes = plt.subplots(ncols=2, nrows=2)
figure.set_size_inches(12, 10)

sns.distplot(house_train["SalePrice"], ax=axes[0][0])
stats.probplot(house_train["SalePrice"], dist='norm', fit=True, plot=axes[0][1])
sns.distplot(np.log(house_trainWithoutOutliers["SalePrice"]), ax=axes[1][0])
stats.probplot(np.log1p(house_trainWithoutOutliers["SalePrice"]), dist='norm', fit=True, plot=axes[1][1])

