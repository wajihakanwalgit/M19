import np as numpy
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix
import pandas as pd


data = pd.read_csv('data.csv')


X = np.arange(10).reshape(-1,1)
y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

model= LogisticRegression(solver='liblinear', c=10.0, random_state=42)


model.fit(X,y)

p_pred= model.predict_proba(X)
y_pred=model.predict(X)
score=model.score(X,y)
conf_m=confusion_matrix(X,y_pred)
report=classification_report(X,y_pred)

print(score)
print(conf_m)
print(report)
print(p_pred)
print(y_pred)
print(model.coef_)
print(model.intercept_)




