import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

data = pd.read_csv("./data.csv")
data.head(10)

X = data.iloc[:, 2:-1]
X

Y = data.iloc[:, 1]
Y = [1 if i=='M' else 0 for i in Y]
Y

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

y_predict = classifier.predict(X_test)
y_predict
cm = confusion_matrix(y_test, y_predict)
sns.heatmap(cm, annot= True)
print(classification_report(y_test, y_predict))

classifier = SVC(kernel = 'linear', random_state = 66)
classifier.fit(X_train, y_train)
y_predict = classifier.predict(X_test)
y_predict
cm = confusion_matrix(y_test, y_predict)
sns.heatmap(cm, annot= True)
print(classification_report(y_test, y_predict))

classifier = SVC(kernel = 'rbf', random_state = 99)
classifier.fit(X_train, y_train)
y_predict = classifier.predict(X_test)
y_predict
cm = confusion_matrix(y_test, y_predict)
sns.heatmap(cm, annot= True)
print(classification_report(y_test, y_predict))
