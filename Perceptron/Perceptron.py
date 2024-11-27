import numpy as np
import pandas as pd
import time
np.random.seed(42)
def read_data(file_path):
    data = pd.read_csv(file_path, header=None)
    X = data.iloc[:, :-1].values  # features, select all rows, all columns except last column
    y = data.iloc[:, -1].values # labels, select all rows, and last column only


    #convert label from 0 and 1 to -1 and 1

    y = np.where(y == 0, -1, 1)
    return X, y


class Perceptron:
    def __init__(self, learning_rate = 0.1, epoch = 10):
        self.learning_rate = learning_rate
        self.epoch = epoch


    def addBias(self, X, num_samples):
        X = np.hstack((np.ones((num_samples, 1)), X))
        return X

class StandardPerceptron(Perceptron):
    def Train(self, X, y):

        # 1) Init weight, dimension == X's:
        num_samples, num_attr = X.shape

        self.weights = np.zeros(num_attr + 1)

        #bias term
        X_biased = self.addBias(X, num_samples)


        # 2) Loop thru epoch
        for epoch in range(self.epoch):

            # i. shuffle the data:
            shuffling_indices = np.random.permutation(num_samples)
            shuffled_X = X_biased[shuffling_indices]
            shuffled_y = y[shuffling_indices]

            # ii. Loop thru each example, update if yiwxi <= 0
            for xi,yi in zip(shuffled_X, shuffled_y):

                if yi * np.dot(self.weights, xi) <= 0:

                    self.weights = self.weights + self.learning_rate * yi * xi

        # 3) Return weights
        return self.weights

    def predict(self, X, w = None):

        X_biased = self.addBias(X, X.shape[0])

        if w is not None:
            return np.sign(np.dot(w, X_biased))


        return np.sign(np.dot(X_biased, self.weights))

    def error(self, X, y):
        predictions = self.predict(X)
        errors = np.mean (predictions != y)
        return errors



class VotedPerceptron(Perceptron):
    def __init__(self, learning_rate = 0.1, epoch = 10):
        super().__init__(learning_rate, epoch)
        self.votes = []



    def Train(self, X, y):
        self.votes = []
        # 1) Init weight, dimension == X's:

        num_samples, num_attr = X.shape

        weights = np.zeros(num_attr + 1)

        # bias term
        X_biased = self.addBias(X, num_samples)

        count = 1
        # 2) For each epoch:
        for epoch in range (self.epoch):
            for xi, yi in zip (X_biased,y):
                if yi * np.dot(weights, xi) <= 0:

                    #save it for later use
                    self.votes.append((weights.copy(), count))
                    weights += self.learning_rate * yi * xi

                    count = 1

                else:
                    count+=1


        self.votes.append((weights.copy(), count))

        return self.votes


    def predict_voted(self, X):

        X_biased = self.addBias(X, X.shape[0])

        prediction = 0
        for w, count in self.votes:
            prediction+= count * np.sign(np.dot(X_biased,w))

        prediction = np.sign(prediction)
        return prediction

    def error_voted(self, X, y):
        predictions = self.predict_voted(X)
        error = np.mean(predictions!=y)
        return error


class AveragedPerceptron(Perceptron):
    def __init__(self, learning_rate = 0.1, epoch = 10):
        super().__init__(learning_rate, epoch)
        self.votes = []
        self.a = None
    def Train(self, X, y):
        num_samples, num_attr = X.shape

        weights = np.zeros(num_attr + 1)

        # bias term
        X_biased = self.addBias(X, num_samples)

        #a
        self.a = np.zeros(num_attr + 1)


        for epoch in range(self.epoch):
            for xi,yi in zip (X_biased, y):
                if yi * np.dot(weights, xi) <= 0:
                    weights+= self.learning_rate * yi * xi

                self.a += weights
        return self.a

    def predict_avg(self, X):
        X_biased = self.addBias(X, X.shape[0])
        return np.sign(np.dot(X_biased, self.a))

    def error_avg(self, X, y):
        predictions = self.predict_avg(X)
        error = np.mean (predictions!=y)
        return error

# Load data
train_X, train_y = read_data('bank-note/train.csv')
test_X, test_y = read_data('bank-note/test.csv')

# Standard Perceptron
standard_perceptron = StandardPerceptron()
w_standard = standard_perceptron.Train(train_X, train_y)
standard_test_error = standard_perceptron.error(test_X, test_y)
print("Learned weight vector:", w_standard)

print("Standard Perceptron Test Error:", standard_test_error)

# Voted Perceptron
voted_perceptron = VotedPerceptron()
w_c = voted_perceptron.Train(train_X, train_y)
voted_test_error = voted_perceptron.error_voted(test_X, test_y)

# Formatting function
def format_array_of_tuples(arr):
    formatted_output = []
    for array, label in arr:
        array_str = np.array2string(array, precision=3, suppress_small=True, max_line_width=60)
        formatted_output.append(f"({array_str}, {label})")
    return "[\n    " + ",\n    ".join(formatted_output) + "\n]"

print("Distinct weight vectors:", format_array_of_tuples(w_c))
print("Voted Perceptron Test Error:", voted_test_error)

# Averaged Perceptron
avg_perceptron = AveragedPerceptron()
learned_Weight_avg = avg_perceptron.Train(train_X, train_y)
avg_test_error = avg_perceptron.error_avg(test_X, test_y)
print("Learned weight vector of Avg:", learned_Weight_avg)

print("Averaged Perceptron Test Error:", avg_test_error)
