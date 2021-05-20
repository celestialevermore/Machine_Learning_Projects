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

# ### 3-1 Accuracy(정확도)

# +
import numpy as np
from sklearn.base import BaseEstimator

class MyDummyClassifier(BaseEstimator):
    
    #fit() 메소드는 아무것도 학습하지 않음.
    def fit(self,x,y=None):
        pass
    
    #predict() 메소드는 단순히 Sex feature가 1이면 0, 그렇지 않으면 1로 예측함
    #입력 데이터에서 feature가 성별 feature가 남자면 0, 여자면 1 -> 이진 분류
    def predict(self,x):
        pred = np.zeros((x.shape[0],1))
        #결정값 pred 
        for i in range(x.shape[0]):
            #남자면, 사망 pred[i]=0
            if x['Sex'].iloc[i]==1:
                pred[i]=0
            #생존하면 pred[i]=1
            else:
                pred[i]=1
        
        return pred



# +
import pandas as pd
from sklearn.preprocessing import LabelEncoder

#Null 처리
def fillna(df):
    df['Age'].fillna(df['Age'].mean(),inplace=True)
    df['Cabin'].fillna('N',inplace=True)
    df['Embarked'].fillna('N',inplace=True)
    df['Fare'].fillna(0,inplace=True)
    
    return df


def drop_features(df):
    df.drop(['PassengerId','Name','Ticket'],axis=1,inplace=True)
    return df

def format_features(df):
    df['Cabin']=df['Cabin'].str[:1]
    features = ['Cabin','Sex','Embarked']
    
    for feature in features:
        le=LabelEncoder()
        le.fit(df[feature])
        df[feature]=le.transform(df[feature])
    return df

#함수 싸악
def transform_features(df):
    df = fillna(df)
    df = drop_features(df)
    df = format_features(df)
    
    return df


    


# +
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#원본 데이터 재로딩, 데이터 가공, 학습데이터, 테스트 데이터 분할까지 
titanic_df=pd.read_csv('C:/Users/김응엽/PerfectGuide-master/PerfectGuide-master/3장/titanic_train.csv')

y_titanic_df = titanic_df['Survived']
X_titanic_df = titanic_df.drop('Survived',axis=1)
X_titanic_df = transform_features(X_titanic_df)

#테스트 분할
X_train,X_test,y_train,y_test = train_test_split(X_titanic_df,y_titanic_df,test_size=0.2,random_state=0)

myclf = MyDummyClassifier()

myclf.fit(X_train,y_train)

mypredictions = myclf.predict(X_test)

print('Dummy Classifier의 정확도:{0:.4f}'.format(accuracy_score(y_test,mypredictions)))





# +
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score

import numpy as np
import pandas as pd

class MyFakeClassifier(BaseEstimator):
    def fit(self, X,y):
        pass
    
    
    #입력값으로 들어오는 X데이터 셋의 크기만큼 전부 0으로 만들어 반환
    def predict(self,X):
        return np.zeros((len(X),1),dtype=bool)
    
#사이킷런 내장 데이터셋인 load_digits를 이용, MNIST 데이터 로딩
digits = load_digits()

print(digits.data)
print('### digits.data.shape:',digits.data.shape)
print(digits.target)
print('### digits.target.shape:',digits.target.shape)

# +
# digit 번호가 7이면 true이고, 이를 astype(int)로, 1로 매핑
# 7이 아닌 나머지는 전부 astype(int)로 0으로 변환
#ndarray 형태로 true false bool값이 리턴됨.
y = (digits.target==7).astype(int)



#훈련, 타겟으로 분류
X_train,X_test,y_train,y_test=train_test_split(digits.data,y,random_state=11)

# +
#q불균형한 레이블 데이터 분포도 확인
print('레이블 테스트셋 크기 :',y_test.shape)
print('테스트셋 레이블 0과 1의 분포도')
print(pd.Series(y_test).value_counts())


#Dummy Classifier로 학습/예측/정확도 평가
fakeclf = MyFakeClassifier()
fakeclf.fit(X_train,y_train)


fakepred = fakeclf.predict(X_test)
print('모든 예측을 0으로 하여도 정확도는 : {:.3f}'.format(accuracy_score(y_test,fakepred)))

# -

# ### Confusion Matrix

# +
from sklearn.metrics import confusion_matrix

#앞에서의 예측 결과인 fakepred와 실제 결과인  y_test의 Confusion_matrix 적용

confusion_matrix(y_test,fakepred)
#근데도 정확도가 90%가 나오는 노릇이니 불균일한 데이터 셋에서는 쓰면 안된다.
# -

# ### 정밀도(Precision) 과 재현율(Recall)

# #### MyFakeClassifier의 예측 결과를 가지고 정밀도, 재현율 측정

# +
from sklearn.metrics import accuracy_score,precision_score,recall_score,confusion_matrix


print('정밀도 :',precision_score(y_test,fakepred))
print('재현율 :',recall_score(y_test,fakepred))
# -

# #### 오차행렬, 정확도, 정밀도, 재현율을 한꺼번에 계산하는 함수

# +
from sklearn.metrics import accuracy_score,precision_score,recall_score,confusion_matrix

def get_clf_eval(y_test,pred):
    confusion = confusion_matrix(y_test,pred)
    accuracy = accuracy_score(y_test,pred)
    precision = precision_score(y_test,pred)
    recall = recall_score(y_test,pred)
    
    print('오차 행렬')
    print(confusion)
    
    print('정확도 : {0:.4f}, 정밀도 : {1:.4f}, 재현율 : {2:.4f}'.format(accuracy,precision,recall))
    


# +
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


#원본 데이터 재로딩, 가공.. 학습/테스트 분리

titani_df = pd.read_csv('C:/Users/김응엽/PerfectGuide-master/PerfectGuide-master/3장/titanic_train.csv')


y_titanic_df = titanic_df['Survived']
X_titanic_df = titanic_df.drop(['Survived'],axis=1)
X_titanic_df = transform_features(X_titanic_df)

X_train,X_test,y_train,y_test = train_test_split(X_titanic_df,y_titanic_df,test_size=0.2,random_state=11)


lr_clf = LogisticRegression()

lr_clf.fit(X_train,y_train)

pred = lr_clf.predict(X_test)
get_clf_eval(y_test,pred)





# -

# ### Precision/Recall Trade-off

# * predict_proba() 메소드 확인

# +
pred_proba =lr_clf.predict_proba(X_test)
pred = lr_clf.predict(X_test)
print('pred_proba() 결과 shape : {0}'.format(pred_proba.shape))
print('pred_proba array에서 앞 3개만 추출\n',pred_proba[3])

# 예측 확률 array와 예측 결과 array를 concatenate -> 예측 확률, 결과값 동시 확인
pred_proba_result = np.concatenate([pred_proba,pred.reshape(-1,1)],axis=1)
print('두 개의 class 중 더 큰 확률을 클래스 값으로 예측 \n',pred_proba_result[:3])
# -


