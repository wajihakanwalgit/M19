import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
dataset = load_breast_cancer()
sns.set_style("whitegrid")  
plt.style.use("fivethirtyeight")
plt.figure(figsize=(10,6))
plt.rcParams.update({'font.size': 18,"figure.figsize":(10,6),"axes.titlepad":22.0})
print("target variable ",dataset['target_name'])
(unique, counts) = np.unique(dataset['target'], return_counts=True)
print("counts ",counts)
print("unique ",unique)
sns.barplot(x=dataset['target_name'],y=counts)
plt.show()
