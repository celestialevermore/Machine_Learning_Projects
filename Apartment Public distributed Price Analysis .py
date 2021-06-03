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
import pandas as pd
import numpy as np

df_last = pd.read_csv("C:/Users/김응엽/Data/주택도시보증공사_전국 평균 분양가격(2019년 12월).csv", encoding="cp949")
#df_last.shape
df_last.head()
# -

df_last.columns

# 해당되는 폴더 혹은 경로의 파일 목록을 출력해 줍니다.
df_first = pd.read_csv("C:/Users/김응엽/Data/전국 평균 분양가격(2013년 9월부터 2015년 8월까지).csv", encoding="cp949")
df_first.head()
df_first.shape
df_first.tail()

df_last.info()

df_last.isnull().sum()

df_last.isna().sum()

df_last["분양가격"]=pd.to_numeric(df_last["분양가격(㎡)"],errors = 'coerce')
df_last["분양가격"]
df_last["분양가격(㎡)"]

df_last["평당분양가격"]=df_last["분양가격"]*3.3
df_last

df_last.info()

df_last["분양가격(㎡)"].describe()

df_last["분양가격"].describe()

df_last["규모구분"].unique()


# +
# df_last["규모구분"].str.replace?
# -

df_last["규모구분"].str.replace

df_last["규모구분"].str.replace("전용면적","")

df_last["전용면적"] = df_last["규모구분"].str.replace("전용면적", "")
df_last["전용면적"]

df_last["규모구분"]

df_last["전용면적"]=df_last["규모구분"].str.replace("전용면적","")
#df_last["전용면적"]
df_last["전용면적"]=df_last["전용면적"].str.replace("초과","~")
#df_last["전용면적"]
df_last["전용면적"]=df_last["전용면적"].str.replace("이하","")
#df_last["전용면적"]
df_last["전용면적"]=df_last["전용면적"].str.replace(" ","").str.strip()
df_last["전용면적"]

df_last
#df_last = df_last.drop(["규모구분","분양가격(㎡)"],axis=1)
df_last.info()

df_last

tmp = pd.DataFrame(df_last.groupby(['지역명'])["평당분양가격"].mean())
tmp
df_last.groupby(['지역명'])["평당분양가격"].describe()

df_last.groupby(["전용면적"])["평당분양가격"].mean()

tmp1 = pd.DataFrame(df_last.groupby(["지역명","전용면적"])["평당분양가격"].mean().round())
tmp1

df_last.groupby(["연도","지역명"])["평당분양가격"].mean().unstack().T
g=df_last.groupby(["연도","지역명"])["평당분양가격"].mean()
g.unstack().transpose()

pd.pivot_table(df_last,index=["지역명"],values=["평당분양가격"],aggfunc="sum")

df_last.groupby(["전용면적"])["평당분양가격"].mean()

df_last.pivot_table(index=["지역명"],values="평당분양가격")

df_last.pivot_table(index = ["전용면적"],columns = ["지역명"],values=["평당분양가격"],aggfunc=np.mean)

# +
tmp2 = df_last.pivot_table(index = ["연도","지역명"],values=["평당분양가격"])

tmp2.loc[2018]

# +
import matplotlib.pyplot as plt

plt.rc("font",family="NanumGothic")

g = df_last.groupby(["지역명"])["평당분양가격"].mean().sort_values(ascending=False)
g.plot(x="지역명",y="평당분양가격",kind = "bar",rot=0,figsize=[10,3])
# -

g.plot.bar(rot=0,figsize=[10,3])

df_last.groupby(["전용면적"])["평당분양가격"].mean().plot(kind="bar",rot=0)

df_last.groupby(["연도"])["평당분양가격"].mean().plot()

p = df_last.pivot_table(index="연도",columns="지역명",values="평당분양가격")
p.plot(kind="box",rot=30,figsize=[15,3])

p.plot(kind = "line",figsize=[10,3],rot=30)

import seaborn as sns
# %matplotlib inline

plt.figure(figsize=(10,3))
sns.barplot(data=df_last,x="지역명",y ="평당분양가격",ci="sd",color="b")

# +
#barplot으로 연도별 평당분양가격

sns.barplot(data=df_last,x="연도",y="평당분양가격")
# -

plt.figure(figsize=(10,3))
sns.lineplot(data=df_last,x="연도",y="평당분양가격",hue="지역명")
plt.legend(bbox_to_anchor=(1.05, 1),loc=2,borderaxespad=0.)

sns.relplot(data=df_last,x="연도",y="평당분양가격",hue="지역명",kind = "line",col="지역명",col_wrap=4,ci=None)

sns.catplot(data=df_last,x="연도",y="평당분양가격",kind="bar",col ="지역명",col_wrap=4)

# +


sns.boxplot(data=df_last, x = "연도",y = "평당분양가격")
# -

plt.figure(figsize=(12,3))
sns.boxplot(data=df_last, x = "연도",y = "평당분양가격",hue="전용면적")

#정규분포곡선과 거의 비슷함
sns.violinplot(data=df_last, x = "연도",y = "평당분양가격")

# ### asd

sns.lmplot(data=df_last,x = "연도",y = "평당분양가격",hue="전용면적",col="전용면적",col_wrap=3)

plt.figure(figsize=(15,3))
sns.swarmplot(data=df_last, x ="연도",y ="평당분양가격",hue="전용면적")

df_last["평당분양가격"].describe()

max_price= df_last["평당분양가격"].max()
max_price

df_last[df_last["평당분양가격"]==max_price]
#조건문을 안에 넣는 경우 ㅇㅇ

# .loc[행]
# .loc[행, 열]
hist1 = df_last["평당분양가격"].hist(bins=100)
hist1
histall = df_last.hist(bins=10)
histall

# +
#결측치가 없는 데이터엣 ㅓ평당분양가격만 가쟈온다.
# .loc[행]
# .loc[행, 열]
# -

price  = df_last.loc[df_last["평당분양가격"].notnull(),"평당분양가격"]
sns.distplot(price)

g = sns.FacetGrid(df_last,row="지역명", height=1.7,aspect=4,)
g.map(sns.distplot,"평당분양가격",hist=False,rug=True)

sns.distplot(price,hist=False, rug=True)

sns.kdeplot(price,cumulative=True)

df_last_notnull=df_last.loc[df_last["평당분양가격"].notnull(),["연도","월","지역명","전용면적"]]
sns.pairplot(df_last_notnull,hue="전용면적")

df_last["전용면적"].value_counts()

df_first

pd.options.display.max_columns=25

df_first

df_last.head()
df_first.head()

df_first.info()

df_first.isnull().sum()

df_first.head(1)

df_first_melt= df_first.melt(id_vars="지역",var_name="기간",value_name="평당분양가격")
df_first_melt.head()

df_first_melt.columns = ["지역명","기간","평당분양가격"]
df_first_melt.head(1)

date = "2013년12월"
date

date_splited= date.split('년')

date_splited[1]

date_splited[0]

date_splited[1].replace("월","")

date_splited


# +
def parse_year(date):
    tmpdate1 = date.split('년')
    #print(tmpdate1)
    newdate=[]
    #ㅋㅋㅋㅋㅋㅋㅋ파이썬 못하겠어
    for i in tmpdate1:
        if i[-1]=='월':
            pass
        else:
            newdate.append(i)
    return int(newdate[0])
            
    
ret = parse_year(date)
ret


# +
def parse_month(date):
    date = date.split('월')
    date = date[0].split('년')
    ret = int(date[1])
    return ret
    
ret = parse_month(date)
ret
# -

df_first_melt["연도"]= df_first_melt["기간"].apply(parse_year)
df_first_melt["연도"]

df_first_melt["월"] = df_first_melt["기간"].apply(parse_month)
df_first_melt["월"]

df_first_melt.head(1)

df_last.head(1)

df_last.columns.tolist()

cols = ["지역명","연도","월","평당분양가격"]
cols

df_last_prepare = df_last.loc[df_last["전용면적"]=="전체",cols].copy()

df_last_prepare.head(1)

df_first_prepare = df_first_melt[cols].copy()
df_first_prepare.head(1)

df = pd.concat([df_first_prepare,df_last_prepare])
df.head()

df["연도"].value_counts(sort = False)

t= pd.pivot_table(df,index = "연도",columns="지역명",values = "평당분양가격").round()
t

# +
plt.figure(figsize=(15,7))

sns.heatmap(t, cmap="Blues", annot = True, fmt=".0f")

# +

t.T
# -

tt = t.T
plt.figure(figsize=(15,7))
sns.heatmap(tt, cmap="Blues", annot = True, fmt=".0f")

df

dd = pd.DataFrame(df.groupby(["연도","지역명"])["평당분양가격"].mean().unstack().round())
dd

plt.figure(figsize=(15,7))
ddd = dd.T
sns.heatmap(ddd, cmap="Greens", annot = True, fmt=".0f")

#barplot
sns.barplot(data=df,x='연도',y='평당분양가격')

# +
#pointplot
plt.figure(figsize=(10,5))

sns.pointplot(data=df,x='연도',y='평당분양가격',hue='지역명')
plt.legend(bbox_to_anchor=(1.05, 1),loc=2,borderaxespad=0.)

# +
# 서울만 barplot으로 그리기
df_seoul = df[df["지역명"]=='서울'].copy()
df_seoul

sns.barplot(data = df_seoul, x = "연도",y = "평당분양가격",color = "b")
sns.pointplot(data = df_seoul, x = "연도",y = "평당분양가격",color = "b")

# +
#boxplot

sns.boxplot(data = df, x="연도",y = "평당분양가격")
# -

sns.boxenplot(data = df, x="연도",y = "평당분양가격")

plt.figure(figsize=(10,4))
sns.violinplot(data = df, x="연도",y = "평당분양가격")

sns.lmplot(data = df, x="연도",y = "평당분양가격")

plt.figure(figsize=(12,5))
sns.swarmplot(data = df, x="연도",y = "평당분양가격",hue="지역명")
plt.legend(bbox_to_anchor=(1.05, 1),loc=2,borderaxespad=0.)

plt.figure(figsize=(12,5))
sns.violinplot(data = df, x="연도",y = "평당분양가격")
sns.swarmplot(data = df, x="연도",y = "평당분양가격",hue="지역명")
plt.legend(bbox_to_anchor=(1.05, 1),loc=2,borderaxespad=0.)

# ### 지역별 분양가를 시각화

plt.figure(figsize=(12,4))
sns.boxplot(data=df,x="지역명",y="평당분양가격")

plt.figure(figsize=(12,4))
sns.boxenplot(data=df,x="지역명",y="평당분양가격")

plt.figure(figsize=(24,4))
sns.swarmplot(data=df,x="지역명",y="평당분양가격")


