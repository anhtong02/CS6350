import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  # or 'Qt5Agg'

# Load the training data
train_data = pd.read_csv('concrete/train.csv', header=None)
test_data = pd.read_csv('concrete/test.csv', header=None)



# Extract X,y from train
X_train = train_data.iloc[:, :-1].values  # columns except slump column
y_train = train_data.iloc[:, -1].values   # slump value



# add col of 1s at the first column
X_train = np.hstack([np.ones((X_train.shape[0], 1)), X_train])

# Extract X,y from test data
X_test = test_data.iloc[:, :-1].values
y_test = test_data.iloc[:, -1].values

# Add col of 1s to X_test
X_test = np.hstack([np.ones((X_test.shape[0], 1)), X_test])


#4a
def gradient_of_cost(X, y, w):
    """
    Gradient of cost
    1/2 * sum_i=1^m (yi - w^Tx_i)^2
    """
    m = len(y)
    predictions = X.dot(w)
    cost = (1 / (2 )) * np.sum((predictions - y) ** 2)
    return cost

def calculate_gradient(X, y, w):
    """
    Compute the gradient for gradient of cost.
    """
    m = len(y)
    predictions = X.dot(w)
    error = predictions - y
    gradient = (1 / m) * X.T.dot(error)
    return gradient

def batch_gradient_descent(X, y, r, decay_rate, epsilon=1e-6, iter_num=1000):
    """Perform batch gradient descent with decaying learning rate."""
    m, n = X.shape
    w = np.zeros(n)  # w is length n for n is attributes (columns) in X
    costs = []

    current_r = r
    for t in range(iter_num):
        # gradient
        gradient = calculate_gradient(X, y, w)

        # apply decaying
        current_r = current_r * decay_rate

        # update weights
        """
        w^t+1 = w^t - r * dj/dw
        """
        w_new = w - current_r * gradient

        # cost
        cost = gradient_of_cost(X, y, w_new)
        costs.append(cost)

        # see if w_new - w has converged
        if np.linalg.norm(w_new - w) < epsilon:
            print(f"Converged, i= {t}, current learning r: {current_r} ")

            break

        # Update weights for next iteration
        w = w_new

    return w, costs


#4b
def sgd(Xi, yi, w):
    """
    Stochastic gradient for a single training example (X_i, y_i).
    """
    prediction = Xi.dot(w)  # w^T x_i == pred
    error = yi - prediction  # y_i - pred
    gradient = error * Xi  # The gradient for all weights with respect to this single example
    return gradient



def stochastic_gradient_descent(X, y, r, decay_rate, tolerance=1e-6, iter_num=500):

    m, n = X.shape
    w = np.zeros(n)
    costs = []

    current_r = r

    for t in range(iter_num):
        for i in range(m):
            # Select one random training example
            X_i = X[i]
            y_i = y[i]

            # Stochastic gradient
            gradient = sgd(X_i, y_i, w)

            # Update weights based on the gradient
            w_new = w + current_r * gradient
            # Check convergence based on the change in weights
            if  np.linalg.norm(w_new - w) < tolerance:
                print("Converged at example = ", i)

                break

            # Update weights for the next iteration
            w = w_new

        current_r = current_r * decay_rate

        cost = gradient_of_cost(X, y, w)
        costs.append(cost)

        #check to see if the past 2 costs already converge, if yes then break
        if len(costs) > 1 and np.abs(costs[-1] - costs[-2]) < tolerance:
            print("already converged at iteration = " , t)
            break

    return w, costs


def analytical_sln(X, y):
    """Calculate the analytical solution for linear regression."""
    return np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)





max_iters = 500
r = 0.7
decay_rate = 0.3

learned_weight, list_of_cost_batch = batch_gradient_descent(X_train, y_train, r, decay_rate, iter_num=max_iters)

# Perform stochastic gradient descent
sgd_learned_weight, list_of_cost_sgd = stochastic_gradient_descent(X_train, y_train, r, decay_rate, iter_num=max_iters)

fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# Plot for Batch Gradient Descent
axs[0].plot(range(len(list_of_cost_batch)), list_of_cost_batch, color='purple')
axs[0].set_title('Cost function Batch')
axs[0].set_xlabel('I')
axs[0].set_ylabel('Cost')

# Plot for Stochastic Gradient Descent
axs[1].plot(range(len(list_of_cost_sgd)), list_of_cost_sgd, color='green')
axs[1].set_title('Cost function SGD')
axs[1].set_xlabel('I')
axs[1].set_ylabel('Cost')

plt.tight_layout()
# Calculate the optimal weights
w_optimal = analytical_sln(X_train, y_train)
print("Learned weight vector of Batch:", learned_weight)
print("Learned weight vector of SGD:", sgd_learned_weight)
print("Optimal weight vector using analytical sln:", w_optimal)

test_cost_optimal = gradient_of_cost(X_test, y_test, w_optimal)



# Calculate the cost on the test data for both methods
test_cost_batch = gradient_of_cost(X_test, y_test, learned_weight)
test_cost_sgd = gradient_of_cost(X_test, y_test, sgd_learned_weight)
print("Test cost using Batch Gradient Descent:", test_cost_batch)
print("Test cost using Stochastic Gradient Descent:", test_cost_sgd)
print("Test cost using analytical sln:", test_cost_optimal)


plt.show()


