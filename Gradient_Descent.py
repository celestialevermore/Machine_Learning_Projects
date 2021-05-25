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

# ### Gradient Descent

# * 실제값을 Y=4x+6 시뮬레이션하는 데이터 값 생성
# * w0 = 6, w1 = 4로 각각 가정하고 시작

# +
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline

np.random.seed(0)


# y = 4x + 6 식을 근사, random 값은 Noise를 윟 ㅐ만듬

X = 2 * np.random.rand(100,1)
y = 6 + 4* X+np.random.randn(100,1)

#X,y 데이터셋 scatter plot으로 시각화
plt.scatter(X,y)
# -

X.shape


def get_weight_updates(w1,w0, X,y , learning_rate=0.01):
    N =len(y)
    
