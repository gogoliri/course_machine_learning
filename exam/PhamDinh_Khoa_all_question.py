# Author: Khoa Pham Dinh
# email: khoa.phamdinh@tuni.fi

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
import pickle
from sklearn import neighbors
from sklearn.tree import DecisionTreeClassifier, plot_tree


def load_data(filename):
    data = np.loadtxt(filename)
    return data


x_tr = load_data("X_train.dat")
y_tr = load_data("y_train.dat")
x_test = load_data("X_test.dat")
y_test = load_data("y_test.dat")


# Define an accuracy function
def accuracy(y, y_hat):
    N = len(y)
    return (y == y_hat).sum() / N


def random_classifier(y):
    classes = len(np.unique(y))
    N = len(y)
    y_hat = np.random.randint(classes, size=N)
    return y_hat


y_hat = random_classifier(y_test)

ran_acc = accuracy(y_test, y_hat)
print(f"Random accuracy: {ran_acc}")

for k in range(1, 6):
    clf = neighbors.KNeighborsClassifier(n_neighbors=k, algorithm="kd_tree", n_jobs=-1)
    clf.fit(x_tr, y_tr)
    y_hat = clf.predict(x_test)
    knn_acc = accuracy(y_test, y_hat)
    print(f"k-NN accuracy {knn_acc} for k={k}")

for k in range(1, 6):
    clf = DecisionTreeClassifier(max_depth=k)
    clf.fit(x_tr, y_tr)
    y_hat = clf.predict(x_test)
    dcs_acc = accuracy(y_test, y_hat)
    print(f"Decision tree accuracy: {dcs_acc} for k={k}")

fig = plt.figure(figsize=(20,20))
clf = DecisionTreeClassifier(max_depth=3)
clf.fit(x_tr, y_tr)
_ = plot_tree(clf)
fig.savefig("PhamDinh_Khoa_tree.png")