# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# ### 사이킷런을 이용하여 붓꽃 데이터 품종 예측하기

# 사이킷런 버전 확인
import sklearn
print(sklearn.__version__)

# #### 붓꽃 예측을 위한 사이킷런 필요 모듈 로딩

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# #### 데이터셋 로딩

# +
import pandas as pd

#붓꽃 데이터셋을 로딩합니다.
iris = load_iris()

iris_data = iris.data

iris_label = iris.target

print('iris target값 : ',iris_label)
print('iris target명 : ',iris.target_names)


#붓꽃 데이터셋을 자세히 보기 위해 DataFrame으로 변환
iris_df = pd.DataFrame(data=iris_data, columns=iris.feature_names)
iris_df['label']=iris.target
iris_df.head(3)
# -

# #### 학습 데이터와 테스트 데이터셋으로 분리
# * X_train : train용 feature셋
# * X_test : test용 feature셋
# * y_train : train용 target셋
# * y_test : test용 target셋

X_train, X_test, y_train, y_test = train_test_split(iris_data, iris_label, 
                                                    test_size=0.2, random_state=11)

# #### 학습 데이터셋으로 학습(train)수행

# +
#DecisionTreeClassifier 객체
dt_clf=DecisionTreeClassifier(random_state=11)


#학습 수행
dt_clf.fit(X_train, y_train)
#학습용 feature 데이터셋에 대하여 결정 데이터셋은 y_train이다.

# -

# #### 테스트 데이터셋으로 예측 수행

#학습이 완료된 DecisionTreeClassifier객체에서 테스트 데이터셋으로 예측수행
pred =dt_clf.predict(X_test)

pred

# #### 예측 정확도 평가

from sklearn.metrics import accuracy_score
print('예측 정확도: {0:.4f}'.format(accuracy_score(y_test,pred)))

# #### 사이킷런 내장 예제 데이터

iris_data = load_iris()
print(type(iris_data))

keys = iris_data.keys()
print('붓꽃 데이터셋의 키들 : ',keys)

att = iris_data.feature_names
print(att)

# 키는 보통 data,target,target_name,feature_names,DESCR로 구성
# * data : 피처의 데이터셋
# * target : 분류에서는 레이블/클래스의 값 회귀에서는 숫자 결과값
# * target_names : 개별 레이블의 이름들
# * feature_names : 속성들의 이름들
# * DESCR : 데이터셋과 각 피쳐의 설명

iris_data.feature_names
iris_data.target_names

# ### Model Selection
# * 학습 데이터셋과 테스트 데이터셋의 분리 - train_test_split()

# +
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


iris_data = load_iris()

dt_clf=DecisionTreeClassifier()
train_data = iris_data.data
train_label = iris_data.target

dt_clf.fit(train_data,train_label)


# 학습 데이터 셋으로 예측 수행
pred = dt_clf.predict(train_data)
print('예측 정확도:',accuracy_score(train_label,pred))

# +
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split



dt_dlf=DecisionTreeClassifier()
iris_data=load_iris()

X_train, X_test,y_train,y_test = train_test_split(iris_data.data,iris_data.target,test_size=0.3,random_state=121)

# -

dt_clf.fit(X_train,y_train)
pred = dt_clf.predict(X_test)
print(accuracy_score(y_test,pred))

# #### 넘파이 ndarray 뿐 아니라 판다스 DataFrame/Series도 train_test_split() 가능

# +
import pandas as pd


iris_df=pd.DataFrame(iris_data.data,columns=iris_data.feature_names)
iris_df['target']=iris_data.target
iris_df.head()

# +
ftr_df = iris_df.iloc[:, :-1] #행은 전체, 열은 -1즉 맨 마지막을 제외
tgt_df=iris_df.iloc[:,-1]

X_train,X_test,y_train,y_test = train_test_split(ftr_df,tgt_df,test_size=0.3,random_state=121)
# -

print(type(X_train),type(X_test),type(y_train),type(y_test))

dt_clf = DecisionTreeClassifier()
dt_clf.fit(X_train,y_train)
pred = dt_clf.predict(X_test)
print(accuracy_score(y_test,pred))

# ### 교차 검증

# * K 폴드

# +
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import numpy as np

iris =load_iris()
features = iris.data
label=iris.target

dt_clf = DecisionTreeClassifier(random_state=156)

#5개의 포ㅓㄹ드 세트로 분리하는 KFOld 객체와 폴드 세트별정확도를 담을 리스트 객체


kfold = KFold(n_splits=5)
cv_accuracy=[] # 경우의 수가 5개 생기고 이를 추합할 예정
print('붓꽃 데이터셋 크기:',features.shape[0])
#이 떄 학습용 데이터셋은 120, 검증용 데이터셋은 30임 왜냐하면 5로 나누었으니깐

# +
n_iter=0


#KFold객체의 split을 호출하면, 폴드 별 학습,검증용 테스트의 로우 인덱스를 array로 반환
#위치 인덱스를 반환
#예를 들어 학습용 데이터셋이 0,1,2,3이고 검증이 4면
#split 결과 0,1,2,3 //// 4로 각각반환


for train_index,test_index in kfold.split(features):
    
    X_train,X_test=features[train_index],features[test_index]
    #학습용 feature셋, 테스트용 feature셋
    y_train,y_test =label[train_index],label[test_index]
    # 학습용 target셋, 테스트용 target셋 반환
    
    #학습 및 에측
    dt_clf.fit(X_train,y_train)
    pred = dt_clf.predict(X_test)
    n_iter+1
    
    #반복 할 때마다의 정확도를 측정
    accuracy = np.round(accuracy_score(y_test,pred),4)
    train_size = X_train.shape[0]
    test_size= X_test.shape[0]
    
    print('\n#{0} 교차 검증 정확도 : {1}, 학습 데이터 크기 : {2}, 검증 데이터 크기: {3}'
         .format(n_iter,accuracy,train_size,test_size))
    print('#{0} 검증 세트 인덱스:{1}'.format(n_iter,test_index))

    #이 과정을 통해 나온 결과를 저장
    cv_accuracy.append(accuracy)

print('\n## 평균 검증 정확도:',np.mean(cv_accuracy))
    
    
# -

# * Strafified K 폴드

# +
import pandas as pd

iris = load_iris()


iris_df=pd.DataFrame(data=iris.data,columns = iris.feature_names)
iris_df['label']=iris.target
iris_df['label'].value_counts()








# +
kfold =KFold(n_splits=3)
# kfold.split(X)는 폴드셋을 5번 반복할 때마다 달라지는 학습/테스트 용 데이터 로우 인덱스 번호 반환
n_iter=0



for train_index, test_index in kfold.split(iris_df):
    n_iter+=1
    label_train = iris_df['label'].iloc[train_index]
    label_test = iris_df['label'].iloc[test_index]
    print('## 교차 검증 : {0}'.format(n_iter))
    print('학습 레이블 데이터 분포 :\n',label_train.value_counts())
    print('검증 레이블 데이터 분포 :\n',label_test.value_counts())

# +
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=3)
n_iter=0

#y값이 들어가야, 즉 결정값 iris_df['label']이 들어가야함. 
for train_index, test_index in skf.split(iris_df,iris_df['label']):
    n_iter +=1 
    label_train = iris_df['label'].iloc[train_index]
    label_test = iris_df['label'].iloc[test_index]
    
    print('## 교차 검증 : {0}'.format(n_iter))
    print('학습 레이블 데이터 분포:\n', label_train.value_counts())
    print('검증 레이블 데이터 분포:\n',label_test.value_counts())
    
    
    

# +
dt_clf = DecisionTreeClassifier(random_state=156)

skfold = StratifiedKFold(n_splits=3)


n_iter=0
cv_accuracy=[]



#StratifiedKFold 의 split() 호출시 반드시 레이블 데이터셋(y) 추가

for train_index, test_index in skfold.split(features,label):
    #split으로 반환된 인덱스를 이용하여, 학습,검증용 테스트 데이터 추출
    X_train,X_test = features[train_index],features[test_index]
    y_train,y_test = label[train_index],label[test_index]
    
    
    #학습, 예측
    dt_clf.fit(X_train,y_train)
    pred = dt_clf.predict(X_test)
    
    #반복 시 마다 정확도를 측정
    n_iter+=1
    
    accuracy = np.round(accuracy_score(y_test,pred),4)
    #소수점 4자리까지
    train_size=X_train.shape[0]
    test_size = X_test.shape[0]
    
    
    print('\n#{0} 교차 검증 정확도 : {1}, 학습 데이터 크기: {2}, 검증 데이터 크기: {3}'.format(n_iter, accuracy, train_size,test_size))
    
    
    cv_accuracy.append(accuracy)
    
    
# 교차 검증별 정화고 및 평균 정확도 계산
print('\n## 교차 검증별 정확도:',np.round(cv_accuracy,4))
print('## 평균 검증 정확도:',np.mean(cv_accuracy))
    
    
    
# -

# * cross_val_score()
# * 지금까지의 것들을 그냥 모듈화해놓음

# +
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.datasets import load_iris
import numpy as np

iris_data = load_iris()


dt_clf = DecisionTreeClassifier(random_state=156)

data = iris_data.data
label = iris_data.target

#성능 지표는 정확도(accuracy), 교차 검증 세트는 3개
scores = cross_val_score(dt_clf,data,label,scoring='accuracy',cv=3)
print('교차 검증별 정확도:',np.round(scores,4))
print('평균 검증 정확도:',np.round(np.mean(scores),4))


# -

# * GridSearchCV

# +
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score


#데이터를 로딩하고, 학습데이터와 테스트 데이터 분리
iris=load_iris()

X_train,X_test,y_train,y_test = train_test_split(iris.data,iris.target,test_size=0.2,random_state=121)

dtree = DecisionTreeClassifier()

### parameter 들을 dictionary 형태로 
parameters = {'max_depth':[1,2,3],'min_samples_split':[2,3]}


# +
import pandas as pd

# param_grid의 하이퍼 파라미터들을 3개의 train, test set fold롤 나누어 테스트 수행 설정
### refit=True 가 defualt임. True면 가장 좋은 파라미터 설정으로 재학습

grid_dtree = GridSearchCV(dtree, param_grid = parameters, cv=3,refit=True, return_train_score=True)


#붓꽃 Train 데이터로 param_grid의 하이퍼 파라미터들을 순차 학습
grid_dtree.fit(X_train,y_train)


#GridSearchCV 결과는 cv_results_라는 딕셔너리로 저장되고, 이를 DataFrame으로 변환
scores_df = pd.DataFrame(grid_dtree.cv_results_)
scores_df[['params','mean_test_score','rank_test_score','split0_test_score','split1_test_score','split2_test_score']]


# -

grid_dtree.cv_results_

# +
print('GridSearchCV 최적 파라미터:',grid_dtree.best_params_)
print('GridSearchCV 최고 정확도: {0:.4f}'.format(grid_dtree.best_score_))

#refit=True로 설정된 GridSearchCV 객체가 fit을 수행 시 학습이 완료된 Estimator를 내포하고 있으므로
#predict()를 통해 예측 가능
pred = grid_dtree.predict(X_test)
print('테스트 데이터세트 정확도: {0:.4f}'.format(accuracy_score(y_test,pred)))

# +
#GridSearchCV의 refit으로 이미 학습된 estimator 반환
estimator = grid_dtree.best_estimator_

#GridSearchCV의 best_estimator_는 이미 최적 하이퍼 파라미터로 학습됨
pred = estimator.predict(X_test)
print('테스트 데이터세트 정확도: {0:.4f}'.format(accuracy_score(y_test,pred)))

# -

# ### 데이터 인코딩

# * 레이블 인코딩(Label encoding)

from sklearn.preprocessing import LabelEncoder

# +
items = ['TV','냉장고','전자레인지','컴퓨터','선풍기','선풍기','믹서','믹서']

#LabelEncoder를 객체로 생성한 후, fit() 과 transform() 으로 label 인코딩 수행

encoder = LabelEncoder()
encoder.fit(items)
labels = encoder.transform(items)
print(items)
print('인코딩 변환값:',labels)
# -

print('인코딩 클래스:',encoder.classes_)

print('디코딩 원본 값:',encoder.inverse_transform([4,5,2,0,1,1,3,3]))

# * 원-핫 인코딩(One Hot encoding)

# +
from sklearn.preprocessing import OneHotEncoder
import numpy as np

encoder = LabelEncoder()
encoder.fit(items)

labels = encoder.transform(items)

#2차원 데이터롤 변환
labels = labels.reshape(-1,1)


#원-핫 인코딩을 적용
oh_encoder = OneHotEncoder()
oh_encoder.fit(labels)

oh_labels = oh_encoder.transform(labels)

print('원-핫 인코딩 데이터')
print(oh_labels.toarray())
print('원-핫 인코딩 데이터 차원')
print(oh_labels.shape)

# +
import pandas as pd

df = pd.DataFrame({'item':['TV','냉장고','전자렌지','컴퓨터','선풍기','선풍기','믹서','믹서']})

df
# -

pd.get_dummies(df)
#바로 이를 수행해줌 ㅇㅇ

# ### 피쳐 스케일링과 정규화

# * StandardScaler

# +
from sklearn.datasets import load_iris
import pandas as pd

#붓꽃 데이터 셋 로딩, DataFrame으로 변환
iris = load_iris()
iris_data =iris.data
iris_df = pd.DataFrame(data = iris_data, columns = iris.feature_names)

print('feature 들의 평균 값:',iris_df.mean())
print('\n feature 들의 분산 값:',iris_df.var())

# +
from sklearn.preprocessing import StandardScaler

#StandardScaler 객체
scaler = StandardScaler()

scaler.fit(iris_df)
iris_scaled = scaler.transform(iris_df)


iris_df_scaled = pd.DataFrame(data=iris_scaled,columns=iris.feature_names)

print('feature들의 평균:',iris_df_scaled.mean())
print('\nfeature들의 분산:',iris_df_scaled.var())
# -

# * MinMaxScaler

# +
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

scaler = MinMaxScaler()
scaler.fit(iris_df)

iris_scaled = scaler.transform(iris_df)

iris_df_scaled = pd.DataFrame(data = iris_scaled, columns = iris.feature_names)

print('feature들의 최소 값:',iris_df_scaled.min())
print('\nfeature들의 최대 값:',iris_df_scaled.max())

# -


