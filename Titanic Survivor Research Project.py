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

# +
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# %matplotlib inline

titanic_df = pd.read_csv('C:/Users/김응엽/PerfectGuide-master/PerfectGuide-master/1장/titanic/titanic_train.csv')
titanic_df.head(3)
# -

# #### 개요
# * Passengerid : 탑승자 데이터 일련번호
# * survived : 생존여부 0 = 사망 1 = 생존
# * Pclass : 티켓의 선실 등급, 1 = 1등석 2 = 2등석 3 = 3등석
# * sex : 탑승자 성별
# * name : 탑승자 이름
# * Age : 탑승자 나이
# * sibsp : 동승 형제자매 혹은 배우자 인원수
# * parch : 동승 부모 혹은 유아 수
# * ticket : 티켓 번호
# * fare : 요금
# * cabin : 선실 번호
# * embarked : 중간 정착 항수 C = Cherbourg, Q = Queenstown, S = Southampton
#

print('\n ### train 데이터 정보 ### \n')
print(titanic_df.info())

# #### NULL 컬럼들에 대한 처리

# +
titanic_df['Age'].fillna(titanic_df['Age'].mean(),inplace=True)
titanic_df['Cabin'].fillna('N',inplace=True)
titanic_df['Embarked'].fillna('N',inplace=True)

print('데이터 세트 Null 개수:',titanic_df.isnull().sum().sum())
# -

print('Sex 값 분포 : \n',titanic_df['Sex'].value_counts())
print('\n Cabin 값 분포 : \n',titanic_df['Cabin'].value_counts())
print('\n Embarked 값 분포 : \n',titanic_df['Embarked'].value_counts())

titanic_df['Cabin'] = titanic_df['Cabin'].str[:1]
print(titanic_df['Cabin'].head(3))

titanic_df.groupby(['Sex','Survived'])['Survived'].count()

sns.barplot(x = 'Sex',y = 'Survived',data = titanic_df)

sns.barplot(x = 'Pclass',y = 'Survived',hue = 'Sex',data = titanic_df)


# +
# 입력 age에 따라 구분값을 반환하는 함수 설정, DataFrame의 apply lambda 식에 사용

def get_category(age):
    cat = ''
    if age<=1:
        cat='Unknown'
    elif age<=5:
        cat = 'Baby'
    elif age <=12:
        cat = 'Child'
    elif age <=18:
        cat = 'Teenager'
    elif age <=25:
        cat = 'Student'
    elif age<=35:
        cat = 'Young adult'
    elif age <=50:
        cat = 'Adlut'
    else:
        cat = 'Elderly'
    
    return cat


# 막대그래프의 크기 figure를 더 크게 설정
plt.figure(figsize=(10,6))

#X축의 값을 순차적으로 표시하기 위해
group_names=['Unknown','Baby','Child','Teenager','Student','Young Adult','Adult','Elderly']

#lambda 식에 위에서 생성한 get_category() 함수를 반환값으로 지정
#get_category(X)는 입력값으로 'Age' 컬럼값을 받아서 해당하는 cat 반환

titanic_df['Age_cat'] = titanic_df['Age'].apply(lambda x : get_category(x))

sns.barplot(x = 'Age_cat',y = 'Survived',hue = 'Sex',data = titanic_df,order=group_names)

titanic_df.drop('Age_cat',axis = 1, inplace =True)





# +
from sklearn import preprocessing

def encode_features(dataDF):
    features = ['Cabin', 'Sex', 'Embarked']
    for feature in features:
        le = preprocessing.LabelEncoder()
        le = le.fit(dataDF[feature])
        dataDF[feature] = le.transform(dataDF[feature])
        
    return dataDF

titanic_df = encode_features(titanic_df)
titanic_df.head()

# +
from sklearn.preprocessing import LabelEncoder

#Null 처리 함수 


def fillna(df):
    df['Age'].fillna(df['Age'].mean(),inplace=True)
    df['Cabin'].fillna('N',inplace = True)
    df['Embarked'].fillna('N',inplace=True)
    df['Fare'].fillna(0,inplace=True)
    return df

#머신러닝 알고리즘에 있어 불필요한 속성 제거

def drop_features(df):
    df.drop(['PassengerId','Name','Ticket'],axis=1,inplace=True)
    return df

#레이블 인코딩 수행
def format_features(df):
    df['Cabin']=df['Cabin'].str[:1]
    features = ['Cabin','Sex','Embarked']
    
    for feature in features:
        le = LabelEncoder()
        le = le.fit(df[feature])
        df[feature]=le.transform(df[feature])
    return df


# 앞에서 설정한 Data Preprocessing 함수 모두를 호출하는 함수
def transform_features(df):
    df = fillna(df)
    df = drop_features(df)
    df = format_features(df)

    return df



    

# +
# 원본 데이터를 재로딩, feature데이터 셋, Label데이터 셋추출
titanic_df = pd.read_csv('C:/Users/김응엽/PerfectGuide-master/PerfectGuide-master/1장/titanic/titanic_train.csv')
y_titanic_df = titanic_df['Survived']
X_titanic_df = titanic_df.drop('Survived',axis=1)

X_titanic_df = transform_features(X_titanic_df)
# -

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_titanic_df,y_titanic_df,test_size=0.2,random_state=11)

# +
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#결정트리, Random Forest, 로지스틱 회귀를 위한 사이킷런 Classifier 클래스 생성
dt_clf = DecisionTreeClassifier(random_state=11)
rf_clf = RandomForestClassifier(random_state=11)
lr_clf = LogisticRegression()

#DecisionTreeClassifier 학습/예측/평가
dt_clf.fit(X_train,y_train)
dt_pred = dt_clf.predict(X_test)

print('DecisionTreeClassifier 정확도 : {0:4f}'.format(accuracy_score(y_test,dt_pred)))

#RandomForestClassifier 학습/예측/평가
rf_clf.fit(X_train,y_train)
rf_pred =rf_clf.predict(X_test)
print('RandomForest 정확도 : {0:4f}'.format(accuracy_score(y_test,rf_pred)))

#LogisticRegression 학습/예측/평가
lr_clf.fit(X_train,y_train)
lr_pred = lr_clf.predict(X_test)
print('LogisticRegression 정확도 : {0:4f}'.format(accuracy_score(y_test,lr_pred)))




# -

from sklearn.model_selection import KFold

