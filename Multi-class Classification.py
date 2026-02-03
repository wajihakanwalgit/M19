import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import linear_model

df =pd.read_csv("https://raw.githubusercontent.com/justmarkham/DAT8/master/data/abtbuy.csv")

features = ['Por','Brittle','Perm','TOC']
target = 'Prod'

X=df[features].values.reshape(-1,len(features))
y=df[target].values

ols= linear_model.LogisticRegression()
model= ols.fit(X,y)

model.coef_

model.intercept_
model.score(X,y)

x_pred=np.array([[12,81,2.31,2.8]])
x_pred=x_pred.reshape(1,len(features))
model.predict(x_pred)

x_pred=np.array([[12,81,2.31,2.8], [15, 60, 2.5, 1]])
x_pred=x_pred.reshape(-
                      1,len(features))
model.predict(x_pred)
