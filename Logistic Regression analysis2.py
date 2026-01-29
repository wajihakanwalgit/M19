from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import pandas as pd
from matplotlib import pyplot as plt

X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42,n_clusters_per_class=1,flip_y=0.03,n_informative=1,n_redundant=0,n_repeated=0)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

plt.scatter(x,y,c=y,cmap='rainbow')
plt.title('Data')   
plt.show()

X_train , X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

log_reg=LogisticRegression()
log_reg.fit(X_train,y_train)

print(log_reg.coef_)
print(log_reg.intercept_)

y_pred=log_reg.predict(X_test)

print(y_pred)
cm=confusion_matrix(y_test,y_pred)
print(cm)