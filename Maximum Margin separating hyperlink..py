import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.datasets import make_blobs


X,Y=make_blobs(n_samples=50, centers=2, random_state=6,cluster_std=0.60)

clf= SGDClassifier(loss="hinge", random_state=42,alpha=0.01, max_iter=200)
clf.fit(X,Y)

xx=np.linspace(-1,5,10)
yy=np.linspace(-1,5,10)
X1,X2=np.meshgrid(xx,yy)
z= np.empty(X1.shape)
for (i,j),val in np.ndenumerate(X1):
    z[i,j]=clf.predict([X1[i,j],X2[i,j]])
plt.contourf(X1,X2,z,cmap="Paired") 

plt.scatter(X[:,0],X[:,1],c=Y,cmap="Paired",edgecolor="k",s=20)
plt.axis("tight")
plt.show()


# Load the breast cancer dataset