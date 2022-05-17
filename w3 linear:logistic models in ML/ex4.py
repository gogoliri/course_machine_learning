# Author: Khoa Pham Dinh
# email: khoa.phamdinh@tuni.fi
import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model


# Load data
def load_xdata(filename):
    with open(filename, 'rb') as file:
        xdata = np.genfromtxt(file)
        return xdata


def load_ydata(filename):
    with open(filename, 'rb') as file:
        ydata = np.genfromtxt(file).astype(np.int32)
        return ydata


# Load the data files
x = load_xdata("X.dat")
y = load_ydata("y.dat")

# normalize x by substract mean
mean = np.mean(x, axis=0)
x_n = (x - mean)  # normalize x by substract the mean

# Problem 1
# Implement logistic regression with sklearn
clf = sklearn.linear_model.LogisticRegression(fit_intercept=False, penalty="none")
clf.fit(x_n, y)
# Save the coefficient
w_sk = clf.coef_.T


# Define an accuracy function
def accuracy(y, y_hat):
    N = len(y)
    return (y == y_hat).sum() / N


# Accuracy of sklearn classifier
y_hat = clf.predict(x_n)
acc_sk = accuracy(y, y_hat)


# Problem 2
# define the function logsig for w and x
def logsig(w, x):
    wx = x @ w
    result = 1 / (1 + np.exp(-wx))
    return result


# Initialize w0
w0 = np.array([[1], [-1]])
N = len(y)

# SSE
# Initialize list to store value at each iteration
w1_sse = []
w2_sse = []
w_sse = w0
acc_sse = []
u = 0.001
# Lsse = sum[(y - (2*logsig(wx) - 1))^2]

# Store value at iteration 0
w1_sse.append(w_sse[0])
w2_sse.append(w_sse[1])
y_sse = 2 * np.around(logsig(w_sse, x_n)) - 1
acc_sse.append(accuracy(y, y_sse[:, 0]))

for i in range(100):
    # Calculate the gradient descent
    a = 2 * (y - 2 * logsig(w_sse, x_n).T + np.ones((1, N))).T * 2 * (-logsig(w_sse, x_n)) * (1 - logsig(w_sse, x_n))
    w_sse = w_sse - u * (a.T @ x_n).T
    # Store value at each iteration
    w1_sse.append(w_sse[0])
    w2_sse.append(w_sse[1])
    y_sse = 2 * np.around(logsig(w_sse, x_n)) - 1
    acc_sse.append(accuracy(y, y_sse[:, 0]))

# Problem 3
# Maximum likelihood
# Initialize list to store value at each iteration
w1_ml = []
w2_ml = []
u = 0.001
w_ml = w0
acc_ml = []
yi = (y == 1)  # Change label -1 and 1 to 0 and 1
# L_ml

# Store value at iteration 0
w1_ml.append(w_ml[0])
w2_ml.append(w_ml[1])
y_ml = 2 * np.around(logsig(w_ml, x_n)) - 1
acc_ml.append(accuracy(y, y_ml[:, 0]))

for i in range(100):
    # Calculate the gradient ascent
    a = (yi - logsig(w_ml, x_n).T).T
    w_ml = w_ml + u * (a.T @ x_n).T
    # Store value at each iteration
    w1_ml.append(w_ml[0])
    w2_ml.append(w_ml[1])
    y_ml = 2 * np.around(logsig(w_ml, x_n)) - 1
    acc_ml.append(accuracy(y, y_ml[:, 0]))

# Plotting
fig, axs = plt.subplots(2, 1)
fig.set_size_inches(18.5, 10.5, forward=True)

# Plot the optimization path
# Plot the weight of SSE and ML
axs[0].plot(w1_ml, w2_ml, "r*-")
axs[0].plot(w1_sse, w2_sse, "b*-")
# Plot the weight of sklearn clf
axs[0].plot(w_sk[0], w_sk[1], "kx")
axs[0].set_xlim([-1, 1.5])
axs[0].set_ylim([-1, 4])
axs[0].grid()
axs[0].set_xlabel("w1")
axs[0].set_ylabel("w2")
axs[0].set_title("Optimization path")
axs[0].legend(["ML rule classification path", "SSE rule classification path", "SKLearn"])

# Plot the accuracy
axs[1].plot(acc_ml, "r")
axs[1].plot(acc_sse, "b")
axs[1].axhline(acc_sk, color='k', linestyle='--')
axs[1].legend(["ML rule classification path accuracy", "SSE rule classification path accuracy", "SKLearn"])
axs[1].grid()
axs[1].set_title("Accuracy at each iteration")
axs[1].set_xlabel("Iteration")
axs[1].set_ylabel("Accuracy(%)")

# Save
fig.savefig('ex4plot.jpg', dpi=100)