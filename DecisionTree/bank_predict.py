import pandas as pd
import numpy as np
from DecisionTree import decision_tree
from DecisionTree.decision_tree import DecisionTree

#load in the datasets for car
bank_labels = ["yes", "no"]
bank_columns = ['age', 'job', 'marital', 'education',
                'default', 'balance', 'housing', 'loan',
                'contact', 'day', 'month', 'duration',
                'campaign', 'pdays', 'previous', 'poutcome', 'y']

numeric_attr = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']


train_data = pd.read_csv('bank_data/train.csv', names= bank_columns)
test_data = pd.read_csv('bank_data/test.csv', names = bank_columns)


#convert numeric to binary values:
def convert_numeric_to_binary(data, attrs):
    data= data.copy()
    for attr in attrs:
        median = data[attr].median()
        data[attr] = np.where(data[attr] <= median, f"<= {median}", f"> {median}")
    return data

gain_methods = ['information_gain', 'majority_error', 'gini']


#A dictionary so can turn to dataframe later.
results = {
    'Depth': [],
    'Method': [],
    'Train Error': [],
    'Test Error': []
}
#for missing val:
def fill_missing_values(data, attributes):
    data = data.copy()
    for attr in attributes:
        majority_val = data[attr][data[attr] != 'unknown'].mode()[0]  # Find the most frequent value
        data[attr] = data[attr].replace("unknown", majority_val)
    return data

#for 3b, UNCOMMENT FOR 3b
fill_vals_train = fill_missing_values(train_data, bank_columns)

#if doing 3b, instead of "train_data", fill in "fill_vals_train" as first param below this line"
revised_train_data = convert_numeric_to_binary(fill_vals_train, numeric_attr)
revised_test_data = convert_numeric_to_binary(test_data, numeric_attr)

for depth in range(1, 17):
    for method in gain_methods:
        tree = DecisionTree(revised_train_data, 'y', attributes=bank_columns[:-1], gain_method=method, max_depth=depth)

        train_predict = decision_tree.predict(tree, revised_train_data)
        test_predict = decision_tree.predict(tree, revised_test_data)

        # Calculate errors
        train_error = decision_tree.calculate_error(train_predict, revised_train_data['y'])
        test_error = decision_tree.calculate_error(test_predict, revised_test_data['y'])

        results['Depth'].append(depth)
        results['Method'].append(method)
        results['Train Error'].append(train_error)
        results['Test Error'].append(test_error)
    print("done depth: ", depth)
# Convert results to a DataFrame and display
results_df = pd.DataFrame(results)
print(results_df)