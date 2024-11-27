import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from scipy.optimize import minimize
import matplotlib.pyplot as plt


# Load Data
def load_data(train_file, test_file):
    train = pd.read_csv(train_file, header=None)
    test = pd.read_csv(test_file, header=None)

    # Split features and labels
    X_train, y_train = train.iloc[:, :-1].values, train.iloc[:, -1].values
    X_test, y_test = test.iloc[:, :-1].values, test.iloc[:, -1].values

    # Convert labels to ±1
    y_train = np.where(y_train == 0, -1, 1)
    y_test = np.where(y_test == 0, -1, 1)

    # Augment features by adding a bias term (last column)
    X_train = np.hstack((X_train, np.ones((X_train.shape[0], 1))))
    X_test = np.hstack((X_test, np.ones((X_test.shape[0], 1))))

    return X_train, y_train, X_test, y_test


# 2)
class PrimalSVM:
    def __init__(self, C, T, schedule, gamma_params):
        self.C = C
        self.T = T
        self.schedule = schedule
        self.gamma_params = gamma_params
        self.w = None
        self.objective_values = []

    def train(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        t = 0

        for epoch in range(self.T):
            X, y = shuffle(X, y)  # Shuffle data for each epoch

            for i in range(n_samples):
                # Learning rate schedule
                gamma_t = self.schedule(t, **self.gamma_params)
                t += 1

                # Update
                if y[i] * np.dot(self.w, X[i]) <= 1:
                    w0 = np.hstack((self.w[:-1], 0))
                    self.w = self.w - gamma_t * w0 + gamma_t * self.C * n_samples * y[i] * X[i]
                else:
                    self.w[:-1] *= (1 - gamma_t)  # Only penalize weights, not bias

            # Compute objective function value
            H_loss = np.maximum(0, 1 - y * np.dot(X, self.w)).sum()
            obj_val = 0.5 * np.dot(self.w[:-1], self.w[:-1]) + self.C * H_loss
            self.objective_values.append(obj_val)

    def predict(self, X):
        return np.sign(X.dot(self.w))

    def error(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions != y)

    def get_objective_curve(self):
        return self.objective_values


# gamma schedule
def gamma_s1(t, gamma_0, a):
    """
    Schedule of learning rate: gamma_t = gamma0 / (1 + (gamma0/alpha) * t)
    :param t: step
    :param gamma_0: initial learning rate
    :param a: alpha
    :return: learning rate at step t, followed schedule 1.
    """
    return gamma_0 / (1 + (gamma_0 / a) * t)


def gamma_s2(t, gamma_0):
    """
    Schedule of learning rate: gamma_t = gamma0 / (1 + t)
    :param t:
    :param gamma_0:
    :return: learning rate a step t, followed schedule 2.
    """
    return gamma_0 / (1 + t)


def dual_objective(alpha, Q):
    """
    Objective following dual form. 0.5∑∑α_iα_jy_iy_j⟨x_i,xj⟩ - ∑α_i.
    Subject to : 0<= a_i <= C for all i;
                ∑α_iy_i=0.

    :param alpha:
    :param Q: = y_iy_j*k(x_i, x_j)
    :return: dual objective
    """
    return -np.sum(alpha) + 0.5 * np.dot(alpha, np.dot(Q, alpha))


def linear_kernel(x1, x2, gamma=None):
    return np.dot(x1, x2)


def gaussian_kernel(x1, x2, gamma):
    """
    k (xi, xj) = exp (-||xi - xj||^2 / gamma )
    :param x1:
    :param x2:
    :param gamma:
    :return:
    """
    return np.exp(-np.linalg.norm(x1 - x2) ** 2 / gamma)


def compute_Q(X, y, kernel, gamma=None):
    """
    Calculate the product of y_i * y_j * k(xi, xj)
    :param X:
    :param y:
    :param kernel: kernel function
    :param gamma:
    :return:
    """
    N = len(X)
    Q = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            Q[i, j] = y[i] * y[j] * kernel(X[i], X[j], gamma)
    return Q


def train_dual_svm(X, y, C, gamma=None):
    if gamma is not None:
        Q = compute_Q(X, y, gaussian_kernel, gamma)
    else:
        Q = compute_Q(X, y, linear_kernel)

    # Constraints and bounds
    constraints = [{'type': 'eq', 'fun': lambda alpha: np.dot(alpha, y)}]
    bounds = [(0, C) for item in range(len(X))]  # make bounds for each alpha_i
    alpha0 = np.zeros(len(X))

    # Solve optimization
    result = minimize(
        fun=dual_objective,
        x0=alpha0,
        args=(Q,),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    return result.x


def get_w_b(X, y, alpha, C, epsilon=1e-5):
    scaled_yi = alpha * y
    w = np.sum(scaled_yi[:, None] * X, axis=0)  # Recover w

    support_vectors = []

    for a in alpha:
        if a > epsilon and a < C - epsilon:
            support_vectors.append(True)
        else:
            support_vectors.append(False)

    examples = y[support_vectors] - np.dot(X[support_vectors], w)
    b = np.mean(examples)
    return w, b


def dual_predict(X, w, b):
    return np.sign(np.dot(X, w) + b)


def dual_error(X, y, w, b):
    predictions = dual_predict(X, w, b)
    return np.mean(predictions != y)


X_train, y_train, X_test, y_test = load_data('bank-note/train.csv', 'bank-note/test.csv')
Cs = [100 / 873, 500 / 873, 700 / 873]
T = 100


def primal_vs_dual(X_train, y_train, X_test, y_test, C_values, T):
    results = {}

    # Train and compare for each C
    for C in C_values:
        # Primal SVM
        for _name, schedule_fn, params in [
            ("Schedule 1", gamma_s1, {"gamma_0": 0.1, "a": 1}),
            ("Schedule 2", gamma_s2, {"gamma_0": 0.1}),
        ]:
            print(f"Training Primal SVM, C={C}, {_name}")
            svm = PrimalSVM(C, T, schedule_fn, params)
            svm.train(X_train, y_train)

            train_error = svm.error(X_train, y_train)
            test_error = svm.error(X_test, y_test)

            results[(C, "Primal", _name)] = {
                "train_error": train_error,
                "test_error": test_error,
                "weights": svm.w,
                "objective_curve": svm.get_objective_curve(),
            }

        # Dual SVM
        print(f"Training Dual SVM, C={C}")
        alpha = train_dual_svm(X_train, y_train, C)
        w, b = get_w_b(X_train, y_train, alpha, C)
        train_error_dual = dual_error(X_train, y_train, w, b)
        test_error_dual = dual_error(X_test, y_test, w, b)

        results[(C, "Dual", None)] = {
            "train_error": train_error_dual,
            "test_error": test_error_dual,
            "weights": w,
            "bias": b,
        }

    # Print results
    for (C, svm_type, schedule), nums in results.items():
        if svm_type == "Primal":
            print(f"{svm_type} SVM C={C}, Schedule={schedule}:")
        else:
            print(f"{svm_type} SVM C={C}:")

        print(f"  Train Error: {nums['train_error']:.4f}")
        print(f"  Test Error: {nums['test_error']:.4f}")
        if svm_type == "Primal":
            print(f"  Weights: {nums['weights']}")
        else:
            print(f"  Weights: {nums['weights']}, Bias: {nums['bias']}")
        print()


def train_dual_svm_kernel(X, y, C, kernel, gamma):
    """
    Train the dual SVM using a kernel.
    """
    n_samples = len(y)
    Q = compute_Q(X, y, kernel, gamma)

    constraints = [{'type': 'eq', 'fun': lambda alpha: np.dot(alpha, y)}]
    bounds = [(0, C) for _ in range(n_samples)]
    alpha0 = np.zeros(n_samples)

    # opt
    result = minimize(
        fun=dual_objective,
        x0=alpha0,
        args=(Q,),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )

    return result.x


def get_bias(X, y, alpha, kernel, gamma, threshold=1e-5):
    """
    Get the bias.
    """
    support_vec = (alpha > threshold) & (alpha < C - threshold)
    if not np.any(support_vec):
        raise ValueError("0 support vectors found.")
    get_sum = np.sum(alpha * y[:, None] * np.array([kernel(X[s], X, gamma) for s in range(len(X))]), axis=1)
    return np.mean(y[support_vec] - get_sum[support_vec])


def dual_predict_with_kernel(X_train, y_train, X_test, alpha, b, kernel, g):
    """
    Predict using the dual SVM with kernel.
    """
    pred = []
    for x in X_test:
        k = np.array([kernel(x, x_train, g) for x_train in X_train])
        get_sum = np.sum(alpha * y_train * k)
        score = get_sum + b
        pred.append(np.sign(score))

    return np.array(pred)


def dual_error_with_kernel(X_train, y_train, X_test, y_test, alpha, b, kernel, g):
    """
    Error rate of dual SVM with kernel.
    """
    predictions = dual_predict_with_kernel(X_train, y_train, X_test, alpha, b, kernel, g)
    return np.mean(predictions != y_test)


def total_support_vectors(alpha, C, epsilon=1e-5):
    return np.sum((alpha > epsilon) & (alpha < C - epsilon))


def overlap_sp(alpha1, alpha2, epsilon=1e-5):
    """
    Count the overlap of support vect between 2 alphas.
    """
    spv1 = np.where((alpha1 > epsilon) & (alpha1 < C - epsilon))[0]
    spv2 = np.where((alpha2 > epsilon) & (alpha2 < C - epsilon))[0]
    return len(np.intersect1d(spv1, spv2))


gammas = [0.1, 0.5, 1, 5, 100]
Cs = [100 / 873, 500 / 873, 700 / 873]

# question 2 and 3a
print("-----------------Now running question 2 and 3a----------------------")
primal_vs_dual(X_train, y_train, X_test, y_test, Cs, T)
print("-----------------DONE--------------------")
print("-----------------Now running question 3b and 3c----------------------")

# question 3b and 3c
support_vectors = {}
for g in gammas:
    for C in Cs:
        # SVM
        alpha = train_dual_svm_kernel(X_train, y_train, C, gaussian_kernel, g)
        b = get_bias(X_train, y_train, alpha, gaussian_kernel, g)

        # Get the errors
        train_error = dual_error_with_kernel(X_train, y_train, X_train, y_train, alpha, b, gaussian_kernel, g)
        test_error = dual_error_with_kernel(X_train, y_train, X_test, y_test, alpha, b, gaussian_kernel, g)

        print(f"Gamma={g}, C={C}: Train Error={train_error:.4f}, Test Error={test_error:.4f}")

        if C == 500 / 873:
            num_support_vectors = total_support_vectors(alpha, C)
            support_vectors[g] = {"alpha": alpha, "num_support_vectors": num_support_vectors}
            print(f"Gamma={g}, C={C}: num of Support Vectors={num_support_vectors}")

for i in range(len(gammas) - 1):
    a1 = support_vectors[gammas[i]]["alpha"]
    a2 = support_vectors[gammas[i + 1]]["alpha"]
    overlap = overlap_sp(a1, a2)
    print(f"Overlap between gamma {gammas[i]} & {gammas[i + 1]} is {overlap}")

print("-----------------DONE--------------------")
