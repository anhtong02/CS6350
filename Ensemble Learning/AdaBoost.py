
from Decision_Tree_hw2 import DecisionTree, predict
import pandas as pd
import numpy as np
import math
import matplotlib
import time

matplotlib.use('Agg')
import matplotlib.pyplot as plt

car_labels = ['unacc', 'acc', 'good', 'vgood']
car_columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'label']




class AdaBoost:
    def __init__(self, T=20):
        self.T = T
        self.alphas = []
        self.trees = []
        self.train_errors = []  # List to store training errors
        self.test_errors = []  # List to store test errors
        self.et=[]
        self.stump_train_errors = []  # Error of each stump on training data
        self.stump_test_errors = []
    def Boosting(self, data1, label, attributes, test_data):
        examples_size = len(data1)
        init_weight = 1 / examples_size
        weights = np.full(examples_size, init_weight)

        data = data1.copy()
        data['weights'] = weights


        for t in range(self.T):

            print("at t =", t)

            tree = DecisionTree(data, label, attributes, max_depth= 1, weight='weights')

            
            self.trees.append(tree)
            train_predict = predict(tree, data1, label)

            #get the stump train error:
            stump_train_error = (train_predict != data1[label]).mean()
            self.stump_train_errors.append(stump_train_error)

            #get the stump test error:
            test_predict = predict(tree, test_data, label)
            stump_test_error = (test_predict != test_data[label]).mean()
            self.stump_test_errors.append(stump_test_error)



            if len(train_predict) != len(data[label]):
                raise ValueError(f"length or predicted labels is not the same as length of original labels column")

            e_t = self.calculate_weighted_error(train_predict, label, data, 'weights')

            self.et.append(e_t)

            epsilon = 1e-10  # Small constant to avoid log(0)

            if e_t == 0:
                print(f"Stopping early at iteration {t} due to perfect classification")
                break
            elif e_t >= 0.5:
                e_t = 0.5 - epsilon

            alpha_t = 0.5 * math.log((1 - e_t) / e_t)
            self.alphas.append(alpha_t)

            for index, (true_label, prediction) in enumerate(zip(data[label], train_predict)):
                if true_label != prediction:  # misclassified
                    data.at[index, 'weights'] *= math.exp(alpha_t)
                else:  # correctly classified
                    data.at[index, 'weights'] *= math.exp(-alpha_t)

            Zt = data['weights'].sum()
            data['weights'] /= Zt  # Normalize

        #train error:
        overall_train_predict = self.predict(data1, label)  # Using the predict function





        #test
        overall_test_predict = self.predict(test_data, label, True)  # Using the predict function


    def predict(self, data, label, test = None):
        # Retrieve unique class labels from the data
        class_labels = data['label'].unique()

        # Initialize an array to store the weighted votes for each class for each data point
        n_examples = len(data)
        n_classes = len(class_labels)
        class_votes = np.zeros((n_examples, n_classes))  # Rows are examples, columns are class labels

        # For each weak classifier (tree) in AdaBoost up to iteration t
        for i in range(len(self.trees)):
            tree = self.trees[i]
            tree_predict = predict(tree, data, label)  # Get predictions from this weak classifier
            alphat = self.alphas[i]


            # For each data point, accumulate the weighted vote
            for j in range(n_examples):
                predicted_label = tree_predict[j]

                # Find the index of the predicted label in the class_labels array
                class_index = np.where(class_labels == predicted_label)[0][0]

                # Add the weighted vote for the predicted class
                class_votes[j][class_index] += alphat


            prediction_at_t = [class_labels[np.argmax(class_votes[i])] for i in range(n_examples)]
            overall_error = np.mean(prediction_at_t != data[label])  # Calculate error

            if test:
                self.test_errors.append(overall_error)

            else:
                self.train_errors.append(overall_error)

        # Final predictions based on the accumulated votes
        final_predictions = [class_labels[np.argmax(class_votes[i])] for i in range(n_examples)]

        return final_predictions

    def calculate_weighted_error(self, predictions, true_labels, data, weight):
        incorrect = predictions != data[true_labels]
        incorrect_weights = data[incorrect][weight]
        return incorrect_weights.sum()



# Run the AdaBoost algorithm

import matplotlib.pyplot as plt


numeric_attr = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']

bank_columns = ['age', 'job', 'marital', 'education',
                'default', 'balance', 'housing', 'loan',
                'contact', 'day', 'month', 'duration',
                'campaign', 'pdays', 'previous', 'poutcome', 'label']


bank_train = pd.read_csv('bank_data/train.csv', names=bank_columns)
bank_test = pd.read_csv('bank_data/test.csv', names=bank_columns)
bank_train.rename(columns={'y': 'label'}, inplace=True)
bank_test.rename(columns={'y': 'label'}, inplace=True)



def convert_numeric_to_binary(data, attrs, test):
    data= data.copy()
    test= test.copy()
    for attr in attrs:
        median = data[attr].median()
        data[attr] = np.where(data[attr] <= median, f"<= {median}", f"> {median}")
        test[attr] = np.where(test[attr] <= median, f"<= {median}", f"> {median}")
    return data, test

bank_train, bank_test = convert_numeric_to_binary(bank_train, numeric_attr, bank_test)



# Run the AdaBoost algorithm
adaboost = AdaBoost(T=500)
adaboost.Boosting(bank_train, 'label', bank_columns[:-1], bank_test)

# Retrieve training and test errors
train_errors = adaboost.train_errors
test_errors = adaboost.test_errors


# First Figure: Overall Training and Test Errors over Iterations
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(train_errors) + 1), train_errors, label='Training Error', color='blue')
plt.plot(range(1, len(test_errors) + 1), test_errors, label='Test Error', color='red')
plt.xlabel('Iteration T')
plt.ylabel('Error Rate')
plt.title('Overall Training and Test Errors over AdaBoost Iterations')
plt.legend()
plt.grid(True)
plt.show()
plt.savefig('f1_AdaBoost.png')
#Get stump errors and plot

stump_train_errors = adaboost.stump_train_errors
stump_test_errors = adaboost.stump_test_errors

# Second Figure: Individual Decision Stump Errors
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(stump_train_errors) + 1), stump_train_errors, label='Stump Training Error', color='green')
plt.plot(range(1, len(stump_test_errors) + 1), stump_test_errors, label='Stump Test Error', color='orange')
plt.xlabel('Iteration T')
plt.ylabel('Error Rate')
plt.title('Training and Test Errors of Individual Decision Stumps')
plt.legend()
plt.grid(True)
plt.show()
plt.savefig('f2_AdaBoost.png')




