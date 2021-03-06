
# 파이썬 리스트는 모든 자료형을 대괄호 안에 넣고 쉼표로 구분한다.
 
# 내부 원소의 순서를 인덱스라 한다 첫번째부터 0,1,2,3...

# 다양한 자료형을 담을 수 있다

x=[1,'삼식이',(1,2,3),6.85]
print('x의 원소종류', x)

# 정의한 이후 해당 인덱스를 불러오거나 수정할 수 있다

x[0]=4
print('인덱스 수정', x)

# 리스트 합치기 곱하기
y=[1,2,3]+[4,5,6]
print('합친 y', y)
y=[4,5,6]*3
print('리스트 반복', y)
# 수식연산
# len 원소갯수
# max 원소중 가장 큰수
# min 원소중 가장 작은 수
# sum 원소합



print('y 원소갯수', len(y))
print('y 원소중 가장 큰수', max(y))
print('가장 작은 수', min(y))
print('원소들의 합 ', sum(y))




# 리스트 내부 항목찾기
print('8' in y)
print('8' not in y)

#같은 메모리를 공유하는 리스트를 수정할 때 주의해야 한다 한쪽을 수정하면 다른 한쪽도 바뀌기 때문이다
y=x
y[0]=10
print('0인덱스가 수정된 y', y)
print('x는 수정하지 않았다 그러나 x=', x)


#append 리스트 뒤에 추가한다 행을 추가한다고 볼 수 있다
x.append(10)
print('x뒤에 10을 붙인다', x.append(10))
#해당하는 항목 숫자를 센다
x.count(10)
print('몇번재에 10이 있는가', x.count(10))

#리스트 뒤에 리스트를 붙인다
x.extend(y)
print('x뒤에 y붙이기', x.extend(y))

#대체가 아니라 뒤로 밀어내면서 입력한다
x.insert(0, 11)
print(x.insert(0, 11))

#오름차순 정렬
#sort
#내림차순
#y.sort(reverse=True)
#슬라이스
#0번째에서 5번째 인덱스까지 불러오기
print(x[0:5] )

#삭제
del(y)
print(y)

# #행렬 5문제

# 1. [[1,2,3], [3,4,2], [2,3,4]] (3,3)
# 2. [6,7] - (2, )
# 3, [[[1,2], [2,4]], [[6,3], [3,2]]] ( 2, 2, 2)
# 4. [3,7,6,8,4,1,2,3,4,2], [1,1,3,2,4,1,5,2,3,2] 
# 5. [283, 123, 345], [123, 434, 345, 123] 


