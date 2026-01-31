from sklearn.linear_model import LogisticRegression
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np

iris=datasets.load_iris()

X=iris.data[:,12]
Y=iris.target

logreg=LogisticRegression(multi_class='multinomial', solver='lbfgs')
logreg.fit(X,Y)

x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5

xx,yy=np.meshgrid(np.arange(x_min, x_max, 0.02),np.arange(y_min, y_max, 0.02))
Z=logreg.predict(np.c_[xx.ravel(), yy.ravel()])

Z=Z.reshape(xx.shape)
plt.figure(1,figsize=(8,6))
plt.pcolormesh(xx,yy,Z,cmap=plt.cm.Paired)

plt.scatter(X[:,0],X[:,1],c=Y,cmap=plt.cm.Paired,edgecolors='k',s=20)
