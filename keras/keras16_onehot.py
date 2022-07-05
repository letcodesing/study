#3가지의 원핫인코딩 방법 비교
from sklearn.preprocessing import OneHotEncoder


get dummie pandas

tensorflow to categorical

sklearn OneHotEncoder

15 covtype 할때 to categorical 로 하면 오류난다 다른 두방식은 ㅇㅋ?


 원핫인코더의 좋은 점은 Ordinal 변수도 모두 범주형으로 인식하여 처리한다는 점이다.
 
 data type은 categorical 명목형과 numerical 순위형으로 나뉜다
 
 명목형 categorical은 nominal ordinal 로 나뉜다
 
numerical은 interval 과 ratio로 나뉜다

from sklearn.preprocessing import OneHotEncoder
​
ohe = OneHotEncoder()
#불러오기
OneHotEncoder문자열 값을 직접 처리 할 수 ​​없습니다. 명목 특징이 문자열 인 경우 먼저 정수로 매핑해야합니다.
pandas.get_dummies그 반대입니다. 기본적으로 열을 지정하지 않는 한 문자열 열만 원-핫 표현으로 변환합니다.
 
 
 문자를 숫자로 바꾸어 주는 방법 중 하나로 One-Hot Encoding이 있다. 

가변수(dummy variable)로 만들어주는 것인데, 이는 0과 1로 이루어진 열을 나타낸다.

1은 있다, 0은 없다를 나타낸다.
[출처] [pandas] pd.get_dummies() :: One-Hot Encoding/원핫인코딩|작성자 밎이


to categorical 은 무조건 0부터 시작하기 때문에 라벨값이 1부터일경우 0, 3부터 시작할경우 0 1 2 를 생성한다
해당하는 데이터전처리가 있다
겟더미와 skleanr은 그냥 숫자만큼만 컬럼을 나눠준다
즉 판다스의 겟더미가 하는 일은 해당 컬럼의 라벨종류별로 숫자를 지정해(아마도 랜덤?) 숫자만큼 컬럼을 만들고 해당하는 라벨컬럼만 True값 즉 1을 주고 나머지는 False = 0값을 주는 array를 만든다
 
 


 