import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

# Load the dataset
data = pd.read_csv('diabetes.csv')

sns.set_style('whitegrid')
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (10, 6)
print("Target variable ",data['target_names'])
(unique, counts) = np.unique(data['Outcome'], return_counts=True)
print('unique values of the target variable',unique )
print('counts of the target variable',counts)
sns.barplot(x=dataset['target_name'],y=counts)
plt.title('Bar plot of target variable')
plt.show()

X=data['data']
y=data['target']

standardrizer= StandardScaler()
X=standardrizer.fit_transform(X)

X_train , X_test , y_train , y_test = train_test_split(X,y,test_size=0.2,random_state=42)
model = LogisticRegression()
model.fit(X_train,y_train)
print('Accuracy of the model is ',model.score(X_test,y_test))

Predicted = model.predict(X_test)

cm=confusion_matrix(y_test,Predicted)

TN,FP,FN,TP=confusion_matrix(y_test,Predicted).ravel()

print('True Posiitive is ',TP)
print('True Negative is ',TN)
print('False Positive is ',FP)
print('False Negative is ',FN)

accuracy = accuracy_score(y_test,Predicted)
print('Accuracy of the model is ',accuracy)


