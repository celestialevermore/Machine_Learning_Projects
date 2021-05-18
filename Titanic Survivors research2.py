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

print(titanic_df.info())

#  ### NULL 컬럼들에 대한 처리

# +
titanic_df['Age'].fillna(titanic_df['Age'].mean(),inplace =True)
titanic_df['Cabin'].fillna('N',inplace=True)
titanic_df['Embarked'].fillna('N',inplace=True)

print(titanic_df.isnull().sum().sum())
# -

print('Sex 분포 : \n',titanic_df['Sex'].value_counts())

titanic_df['Cabin']=titanic_df['Cabin'].str[:1]
print(titanic_df['Cabin'].head(3))

titanic_df.groupby(['Sex','Survived'])['Survived'].count()

sns.barplot(x='Sex',y = 'Survived',data=titanic_df)

sns.barplot(x='Pclass',y='Survived',hue='Sex',data=titanic_df)


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

#막대그래프의 
plt.figure(figsize=(10,6))
#X축의 값을 순차적으로 표시하기 위해
group_names=['Unknown','Baby','Child','Teenager','Student','Young Adult','Adult','Elderly']

#lambda 식에 위에서 생성한 get_category() 함수를 반환값으로 지정
#get_category(X)는 입력값으로 'Age' 컬럼값을 받아서 해당하는 cat 반환


titanic_df['Age_cat']=titanic_df['Age'].apply(lambda x : get_category(x))
sns.barplot(x='Age_cat',y = 'Survived',hue='Sex',data=titanic_df,order=group_names)

titanic_df.drop('Age_cat',axis=1,inplace=True)

# +
from sklearn import preprocessing

def encode_features(dataDF):
    features = ['Cabin','Sex','Embarked']
    for feature in features:
        le = preprocessing.LabelEncoder()
        le=le.fit(dataDF[feature])
        dataDF[feature]=le.transform(dataDF[feature])
    
    return dataDF

titanic_df = encode_features(titanic_df)
titanic_df.head()

# +
from sklearn.preprocessing import LabelEncoder

#Null 처리
def fillna(df):
    df['Age'].fillna(df['Age'].mean(),inplace=True)
    df['Cabin'].fillna('N',inplace=True)
    df['Embarked'].fillna('N',inplace=True)
    df['Fare'].fillna(0,inplace=True)
    
    return df


print(titanic_df.columns)

def drop_features(df):
    df.drop(['PassengerId','Name','Ticket'],axis=1,inplace=True)
    
    return df

def format_features(df):
    df['Cabin']=df['Cabin'].str[:1]
    features=['Cabin','Sex','Embarked']
    for feature in features:
        le=LabelEncoder()
        le=le.fit(df[feature])
        df[feature]=le.transform(df[feature])
        
    return df


#앞에서 설정한 Data Preprocessing 호출


def transfrom_features(df):
    df = fillna(df)
    df = drop_features(df)
    df = format_features(df)
    
    return df






# +
# 원본 데이터 재로딩, feature 데이터셋 <- > label 데이터 셋 추출
titanic_df = pd.read_csv('C:/Users/김응엽/PerfectGuide-master/PerfectGuide-master/1장/titanic/titanic_train.csv')


y_titanic_df = titanic_df['Survived']
X_titanic_df = titanic_df.drop('Survived',axis=1)

X_titanic_df = transfrom_features(X_titanic_df)
# -

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X_titanic_df,y_titanic_df, test_size=0.2,random_state=11)

# +
## 머러 알고리즘 총 3개를 사용하여 표현
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#결정트리, RandomForest, 로지스틱 회귀 -> 사이킷런 Classifier 클래스

dt_clf = DecisionTreeClassifier(random_state=11)
rf_clf = RandomForestClassifier(random_state=11)
lr_clf = LogisticRegression()

#1. Descision으로 평가
dt_clf = dt_clf.fit(X_train,y_train)
dt_pred = dt_clf.predict(X_test)
print('DescisionTreeClassifier 정확도 : {0:.4f}'.format(accuracy_score(y_test,dt_pred)))


#2. RandomForest로 평가
rf_clf = rf_clf.fit(X_train,y_train)
rf_pred = rf_clf.predict(X_test)
print('RandomForestClassifier 정확도 : {0:.4f}'.format(accuracy_score(y_test,rf_pred)))

#3. 로지스틱 회귀로 평가
lr_clf = lr_clf.fit(X_train,y_train)
lr_pred = lr_clf.predict(X_test)
print('LogisticRegression 정확도: {0:.4f}'.format(accuracy_score(y_test,lr_pred)))


# +
from sklearn.model_selection import KFold

def exec_kfold(clf,folds=5):
    #폴드 세트를 5개인 KFold 객체 생성, 폴드 수만큼 예측결과 저장 리스트
    kfold=KFold(n_splits=5)
    scores=[]
    
    #KFold 교차 검증 수행
    for iter_count, (train_index,test_index) in enumerate(kfold.split(X_titanic_df)):
        # X_titanic_df 데이터에서, 교차 검증 인덱스별로 각각 학습,검증 데이터를 가리키는 인덱스 생성
        
        X_train,X_test = X_titanic_df.values[train_index],X_titanic_df.values[test_index]
        y_train,y_test = y_titanic_df.values[train_index],y_titanic_df.values[test_index]

        #Classifier 학습 ,예측, 정확도 계산
        clf.fit(X_train,y_train)
        
        predictions = clf.predict(X_test)
        accuracy = accuracy_score(y_test,predictions)
        scores.append(accuracy)
        print("교차 검증 {0} 정확도: {1:.4f}".format(iter_count,accuracy))
        
    # 5개 fold에서의 평균 정확도 계산
    mean_score = np.mean(scores)
    print("평균 정확도: {0:.4f}".format(mean_score))


#exec_kfold 호출
exec_kfold(dt_clf,folds=5)

# +
from sklearn.model_selection import cross_val_score

scores = cross_val_score(dt_clf,X_titanic_df,y_titanic_df,cv=5)

for iter_count,accuracy in enumerate(scores):
    print("교차 검증 {0} 정확도: {1:.4f}".format(iter_count,accuracy))

print('평균 정확도:{0:.4f}'.format(np.mean(scores)))
# -

# *GridSearchCV를 통한 하이퍼파라미터 튜닝

# +
from sklearn.model_selection import GridSearchCV

params = {'max_depth':[2,3,5,10],
         'min_samples_split':[2,3,5],'min_samples_leaf':[1,5,8]}

grid_dclf = GridSearchCV(dt_clf, param_grid = params, scoring='accuracy',cv=5)
grid_dclf.fit(X_train,y_train)


print('GridSearchCV 최적 하이퍼 파라미터 :',grid_dclf.best_params_)
print('GridSearchCV 최고 정확도: {0:.4f}'.format(grid_dclf.best_score_))
best_dclf = grid_dclf.best_estimator_

#GridSearchCV의 최적 하이퍼 파라미터로 학습된 Estimator로 예측 평가 수행

dpredictions = best_dclf.predict(X_test)
accuracy = accuracy_score(y_test,dpredictions)

print("테스트 세트에서의 DecisionTreeClassifier 정확도 : {0:.4f}".format(accuracy))
# -


