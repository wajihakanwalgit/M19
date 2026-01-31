import numpy as np

class LogisticRegression:
    def fit(self, X, y , lr=0.001 , epochs=10000,verbose=True,batch_size=1):
        self.classes = np.unique(y)
        y=(y==self.classes[1])*1
        X = self.add_bias (X)
        self.weights=np.zeros(X.shape[1])
        self.loss=[]
        for i in range(epochs):
            self.loss.append(self.cross_entropy_loss(X,y))
            if i % 1000 == 0 and verbose:
                print(f'Epoch {i}, Loss: {self.loss[-1]}')
            idx= np.random.choice(X.shape[0], batch_size, replace=False)
            X_batch, y_batch = X[idx],y[idx]
            self.weights -= lr * self.gradient(X_batch, y_batch)
        return self
      


    def add_intercept(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))