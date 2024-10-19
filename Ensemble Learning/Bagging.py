import numpy as np
import pandas as pd
from Decision_Tree_hw2 import DecisionTree, predict
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

from joblib import Parallel, delayed

class Bagging:
    def __init__(self, T=500):
        self.T = T
        self.trees = []
        self.train_errors = []
        self.test_errors = []

    def Bag(self, data, label, attributes, c=None, d=None):
        # Use joblib to parallelize tree building
        self.trees = Parallel(n_jobs=-1)(delayed(self.build_tree)(data, label, attributes, c, d) for t in range(self.T))

    def build_tree(self, data, label, attributes, c=None, d=None):
        # Draw examples
        if c:
            drawn_ex = self.draw_examples(data, c)
        else:
            drawn_ex = self.draw_examples(data)


        # Train a classifier
        if d:
            tree = DecisionTree(drawn_ex, label, attributes, random_forest=d)
        else:
            tree = DecisionTree(drawn_ex, label, attributes)  # Use Info Gain by default
        return tree

    def draw_examples(self, data, c=None):
        m = len(data)
        if c:  # adjust later if want smaller size m'
            indices = np.random.choice(m, size=1000, replace=False)
        else:
            indices = np.random.choice(m, size=m, replace=True)

        return data.iloc[indices].reset_index(drop=True)

    def predict(self, data, label, test=False):
        """
        Predict labels for the data using the ensemble of trees.
        Parallelize tree predictions but aggregate the majority vote sequentially.
        """
        # Step 1: Predict using all trees in parallel
        tree_predictions = Parallel(n_jobs=-1)(delayed(predict)(tree, data, label) for tree in self.trees)

        # Step 2: Aggregate predictions tree-by-tree and calculate error incrementally
        predictions = []
        for t, tree_pred in enumerate(tree_predictions):
            predictions.append(tree_pred)  # Add new tree's predictions
            prediction_at_t = pd.DataFrame(predictions).T  # Transpose to align predictions by data point

            # Step 3: Compute the majority vote up to tree `t`
            majority_vote_at_t = prediction_at_t.mode(axis=1)[0]

            # Step 4: Calculate the error
            error = np.mean(majority_vote_at_t != data[label])

            # Step 5: Store the error in the appropriate list
            if test:
                self.test_errors.append(error)
            else:
                self.train_errors.append(error)

        # Final majority vote prediction after all trees
        majority_vote_predictions = prediction_at_t.mode(axis=1)[0]
        return majority_vote_predictions


        #this is just for all the trees
        predictions_df = pd.DataFrame(predictions).T  #Transpose so each row is a data point, and each column is a tree's prediction

        majority_vote_predictions = predictions_df.mode(axis=1)[0]
        return majority_vote_predictions



bank_columns = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing',
                'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays',
                'previous', 'poutcome', 'label']
numeric_attr = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']

# Read the CSV files
bank_train = pd.read_csv('bank_data/train.csv', names=bank_columns)
bank_test = pd.read_csv('bank_data/test.csv', names=bank_columns)

bank_train.rename(columns={'y': 'label'}, inplace=True)
bank_test.rename(columns={'y': 'label'}, inplace=True)


# Convert numeric columns to binary
def convert_numeric_to_binary(data, attrs, test):
    data= data.copy()
    test= test.copy()
    for attr in attrs:
        median = data[attr].median()
        data[attr] = np.where(data[attr] <= median, f"<= {median}", f"> {median}")
        test[attr] = np.where(test[attr] <= median, f"<= {median}", f"> {median}")
    return data, test

bank_train, bank_test = convert_numeric_to_binary(bank_train, numeric_attr, bank_test)

t=500
print("Running 2b at default t =", t)
bagged_trees = Bagging(T=t)
bagged_trees.Bag(bank_train, 'label', bank_columns[:-1])

train_predictions = bagged_trees.predict(bank_train, 'label')
test_predictions = bagged_trees.predict(bank_test, 'label', test = True)




import matplotlib

# Retrieve training and test errors
train_errors = bagged_trees.train_errors
test_errors = bagged_trees.test_errors


# First Figure: Overall Training and Test Errors over Iterations
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(train_errors) + 1), train_errors, label='Training Error', color='blue')
plt.plot(range(1, len(test_errors) + 1), test_errors, label='Test Error', color='red')
plt.xlabel('Iteration T')
plt.ylabel('Error Rate')
plt.title('Overall Training and Test Errors over Bagging')
plt.legend()
plt.grid(True)
plt.savefig('plot_for_Bagging.png')
print("done 2b")
print("----------------------------------------------------------")



"""-------------------------------------------------------------
For 2c

----------------------------------------------------------------
"""
def question_2c(data, label, attributes, test_data, test_label, repeat = 100, T = 500):
    """
    Repeat 100 times, each time bag 500 trees.
    :param data: train dataframe
    :param label: a name of label/outcome column of train dataframe, this can be a string or int
    :param attributes: features value, an array
    :param test_data: test dataframe
    :param test_label: a name of label/outcome column of test dataframe, can be a string or int
    :return: Does not return anything but will print out the result needed for question 3c
    """
    
    single_predictions = []
    bagged_predictions= []


    for i in range(repeat):
        print(f"at {i + 1} / {repeat}")
        bagged_trees = Bagging(T)
        bagged_trees.Bag(data, label, attributes, c = True)

        # first tree's
        first_tree = bagged_trees.trees[0]
        first_tree_predict = predict(first_tree, test_data, label)
        single_predictions.append(first_tree_predict)

        # Bagged predictions from 500 trees
        bagged_predict = bagged_trees.predict(test_data, label)
        bagged_predictions.append(bagged_predict)

        single_df = pd.DataFrame(single_predictions).T
        bagged_df = pd.DataFrame(bagged_predictions).T

    # Compute bias, variance, and squared error for single trees and bagged trees
        true_labels = test_data[label].map({'yes': 1, 'no': 0})

        single_bias, single_variance, single_squared_error = bias_and_variance(single_df, true_labels)
        bagged_bias, bagged_variance, bagged_squared_error = bias_and_variance(bagged_df, true_labels)

        print(f"at i = {i + 1}:")
        print(
            f"Single tree -> bias: {single_bias}, variance: {single_variance}, squared error: {single_squared_error}")
        print(
            f"Bagged -> bias: {bagged_bias}, variance: {bagged_variance}, squared error: {bagged_squared_error}")



def bias_and_variance(predictions_df, true_labels):
    """
    calculate the bias and variance between predictions and true labels
    :param predictions_df: a series, not df
    :param true_labels: a series
    :return: bias, variance, squared error
    """

    predictions_df = predictions_df.replace({'yes': 1, 'no': 0})

    #avg prediction for each row
    avg_predictions = predictions_df.mean(axis=1)

    true_labels = true_labels.astype(float)



    # bias: (avg_prediction - true_label)^2
    bias_squared = np.mean((avg_predictions - true_labels) ** 2)


    # variance of the predictions
    variance = np.mean(predictions_df.var(axis=1))

    #squared error: bias + variance
    squared_error = bias_squared + variance

    return bias_squared, variance, squared_error


import matplotlib

# c)
# Ask the user if they want to run question 2c
print("Running 2c")
user_input = input("WARNING (it takes a lot of time!!!) : Do you want to run question 2c? (yes/no): ").strip().lower()

run_question_2c = False  

# Default values for number of repeats and number of T
DEFAULT_REPEATS = 100  # default
DEFAULT_T = 500  # Eample default


if user_input == "yes":
    run_question_2c = True  # Set to True if the user says 'yes'


if run_question_2c:
    while True:
        try:
            num_repeats_input = input(f"Please enter the number of repeats (default Repeat= {DEFAULT_REPEATS}). Or press enter for default: ").strip()
            if num_repeats_input == "":
                num_repeats = DEFAULT_REPEATS  
            else:
                num_repeats = int(num_repeats_input)  

            num_T_input = input(f"Please enter the number of T (default T= {DEFAULT_T}). Or press enter for default: ").strip()
            if num_T_input == "":
                num_T = DEFAULT_T  
            else:
                num_T = int(num_T_input)  

            break  

        except ValueError:            
            print("Invalid input. Please enter valid integers for both inputs or press Enter for defaults.")

    question_2c(bank_train, 'label', bank_columns[:-1], bank_test, 'label', num_repeats, num_T)
    print("Done 2c")
    print("--------------------------------------------------------")

else:
    print("Skipping question 3c.")



#d) random forest
def random_forest(data, label, attributes, test_data, test_label):
    T = 500
    subset = [2,4,6]

    # Create a figure with subplots, one for each subset size
    fig, axes = plt.subplots(1, len(subset), figsize=(18, 6), sharey=True)

    for idx, i in enumerate(subset):
        print("Processing subset size:", i)
        bagged_trees = Bagging(T)
        bagged_trees.Bag(data, label, attributes, d=i)
        print("Done bagging:", i)

        print("start predict train:", i)

        train_predictions = bagged_trees.predict(data, label)
        print("done predict train:", i)

        print("start predict test:", i)

        test_predictions = bagged_trees.predict(test_data, test_label, test=True)
        print("done predict test:", i)

        # Retrieve training and test errors
        train_errors = bagged_trees.train_errors
        test_errors = bagged_trees.test_errors

        # Plot training and test errors on the corresponding subplot
        axes[idx].plot(range(1, len(train_errors) + 1), train_errors, label='Training Error', color='blue')
        axes[idx].plot(range(1, len(test_errors) + 1), test_errors, label='Test Error', color='red')
        axes[idx].set_xlabel('Iteration T')
        axes[idx].set_title(f'Subset = {i}')
        axes[idx].grid(True)
        axes[idx].legend()

    # Set common labels and title
    fig.suptitle('Training and Test Errors for Random Forest with Different Subsets')
    fig.text(0.5, 0.04, 'Iteration T', ha='center')
    fig.text(0.04, 0.5, 'Error Rate', va='center', rotation='vertical')

    # Show the entire figure
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to fit the overall title
    plt.show()
    plt.savefig('RandomForest.png')



print("Running random forest at default, T = 500")
random_forest(bank_train, 'label', bank_columns[:-1], bank_test, 'label')
print("-----------------------------------------------------")