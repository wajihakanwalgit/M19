import imblearn
from pandas import read_csv
from collections import Counter
from matplotlib import pyplot
from sklearn.preprocessing import LabelEncoder
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/glass.csv'
df=read_csv(url,header=None)
data=df.values
X,y=data[:,:-1],data[:,-1]
y=LabelEncoder().fit_transform(y)
counter=Counter(y)
for k,v in counter.items():
    percent = v / len(y) * 100
    print('Class=%d, Count=%d, Percentage=%.3f%%' % (k, v, percent))
pyplot.bar(counter.keys(), counter.values())
pyplot.show()

