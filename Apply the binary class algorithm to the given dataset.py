import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import math


data = pd.read_csv('data.csv')

data.head()
plt.scatter(data.age,data.bought_insurance,marker='+',color='red')  

X_train, X_test, y_train, y_test = train_test_split(data[['age']], data.bought_insurance, test_size=0.2, random_state=42)

X_test

model=LogisticRegression()

model.fit(X_train,y_train)

X_test

y_pred=model.predict(X_test)

model.predict_proba(X_test)

model.score(X_test,y_test)

y_pred

model.coef_

model.intercept_

def sigmoid(x):
    return 1/(1+math.exp(-x))

def predict_function(x):
    return sigmoid(model.intercept_+model.coef_*x)

age=35
predict_function(age)

age=43
predict_function(age)
