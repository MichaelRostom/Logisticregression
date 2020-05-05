import numpy as np
from matplotlib import pyplot as plt
import math
import scipy.optimize as opt

# Load Data
Loaded_Data = np.genfromtxt('ex2data1.txt', delimiter=',')  # Get Data from file
X = Loaded_Data[:, 0 :2]  # Load X
y = Loaded_Data[:, 2]  # load y
m = y.shape[0]  # Set The Number OF training examples
n = X.shape[1]  # Sets The Number of features
y = y.reshape((m, 1))  # reshape y into correct shape

# Plot the Data
X_Ad = X[np.where(y > 0)[0], :]  # Get only the admitted students
X_NAd = X[np.where(y < 1)[0], :]  # Get only the NOT admitted students
plt.scatter(X_Ad[:, 0], X_Ad[:, 1], marker='+', color='k',
            label='Admitted Students', )  # plots the admitted students
plt.scatter(X_NAd[:, 0], X_NAd[:, 1], marker='o', color='y',
            label='Not Admitted Students')  # plots the NOT admitted students
plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')
plt.legend()
plt.show()


# Calculate the sigmoid function
def sigmoid(z) :
    g = np.zeros(z.shape)
    g = 1 / (1 + (math.e ** -z))
    return g


# add the intercept form to x
X = np.hstack([np.ones((m, 1)), X])

# Initialize theta
theta = np.zeros((n + 1, 1))


# Calculate the cost function
def Costfunction(X, y, theta) :
    m = y.shape[0]  # set the shape
    j = (-1 / m) * (np.matmul(y.transpose(), np.log(sigmoid(np.matmul(X, theta)))) +
                    np.matmul(1 - (y.transpose()),
                              np.log(1 - (sigmoid(np.matmul(X, theta))))))  # Calculate the cost function
    grad = (1 / m) * np.matmul(X.transpose(), np.subtract(sigmoid(np.matmul(X, theta)), y))  # Calculate the gradient
    return j, grad


# Compute and display cost and gradient with non-zero theta
test_theta = np.array(([-24], [0.2], [0.2]))
cost, grad = Costfunction(X, y, test_theta)


# Find the best theta by using fminunc
result = opt.fmin_tnc(func=cost,
                      x0=theta,
                      fprime=grad,
                      args=(X, y))
print(result)
# Costfunction(result[0], X, y)
