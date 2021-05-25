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

# ### 결정 트리 모델의 시각화(Decision Tree Visualization)

# +
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings('ignore')


#DecisionTreeClassifier 생성
dt_clf = DecisionTreeClassifier(random_state=156)

#붓꽃 데이터를 로딩하고, 학습과 테스트 데이터셋으로 분리
iris_data=load_iris()
X_train,X_test,y_train,y_test = train_test_split(iris_data.data,iris_data.target,test_size=0.2,random_state=11)


#DecisionTreeClassifier학습
dt_clf.fit(X_train,y_train)

# +
from sklearn.tree import export_graphviz

#export_graphvis()의 호출 결과로 out_file로 지정된 tree.dot 파일 생성합니다.

export_graphviz(dt_clf, out_file = 'tree.dot',class_names=iris_data.target_names, feature_names = iris_data.feature_names,impurity=True,filled=True)
# -

# ### 트리 시각화 모듈 지점

# +
import graphviz

#위에서 생성된 tree.dot 파일을 Graphviz 읽어서 Jupyter Notebook 상에서 시각화 시작

with open("tree.dot") as f:
    dot_graph = f.read()
    
graphviz.Source(dot_graph)
# -

# #### 결정 트리의 feature 선택 중요도

# +
import seaborn as sns
import numpy as np
# %matplotlib inline

#feature importance 추출
print("Feature importance : \n{0}".format(np.round(dt_clf.feature_importances_,3)))



#feature 별 importance 매핑
for name, value in zip(iris_data.feature_names, dt_clf.feature_importances_):
    print('{0}:{1:.3f}'.format(name,value))
    
    
#feature importance를 column 별로 시각화
sns.barplot(x=dt_clf.feature_importances_,y=iris_data.feature_names)
# -


