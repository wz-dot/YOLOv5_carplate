import svm as svm
from sklearn import datasets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

number_data=datasets.load_digits()

from sklearn.svm import SVC
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
clf = svm.SVC(decision_function_shape='ovr')
x_train, x_test, y_train, y_test = train_test_split(number_data.data, number_data.target, test_size=0.2)
clf.fit(x_train, y_train)
print(confusion_matrix(y_test, clf.predict(x_test)))
print(accuracy_score(y_test, clf.predict(x_test)))

from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(solver='adam', alpha=1e-5,
                    hidden_layer_sizes=(30,), random_state=1)
x_train, x_test, y_train, y_test = train_test_split(number_data.data, number_data.target, test_size=0.2)
clf.fit(x_train, y_train)
print(confusion_matrix(y_test, clf.predict(x_test)))
print(accuracy_score(y_test, clf.predict(x_test)))

numlist = list()
for i in range(6):
    numlist.append(random.randint(0,1797))
result = np.hstack((number_data.images[numlist[0]],
                number_data.images[numlist[1]],
                number_data.images[numlist[2]],
                number_data.images[numlist[3]],
                number_data.images[numlist[4]],
                number_data.images[numlist[5]]))
plt.gray()
plt.matshow(result)
print(f"随机数据：{number_data.target[numlist[0]]}{number_data.target[numlist[1]]}"
      f"{number_data.target[numlist[2]]}{number_data.target[numlist[3]]}"
      f"{number_data.target[numlist[4]]}{number_data.target[numlist[5]]}")
plt.show()

# SVM方法
from sklearn.svm import SVC
from sklearn import svm
clf = svm.SVC(decision_function_shape='ovr')
clf.fit(number_data.data, number_data.target)
for i in range(6):
    pred = clf.predict(number_data.data[numlist[i]].reshape(1,64))
    print(pred,end='')


