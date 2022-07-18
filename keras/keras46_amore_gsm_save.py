#데이터 불러오기
import pandas as pd
d_s = pd.read_csv('c:/study/_data/test_amore_0718/삼성전자220718.csv',encoding='cp949')
d_a = pd.read_csv('c:/study/_data/test_amore_0718/아모레220718.csv',encoding='cp949')
#삼전 10/03/24부터 모레 09/09/01 
#최대한 많은 양 잘라주기
d_s = d_s.loc[:3041]
d_a = d_s.loc[:3041]
print(d_s.loc[0],d_a.loc[0])
print(d_s.shape, d_a.shape)
#dropcol 정하기
print(d_s.columns)
#모델구성위해 x,y로 나누기
# sx = d_s.