import pandas as pd
import numpy as np
import decision_tree

#load in the datasets for car
bank_labels = ["yes", "no"]
bank_columns = ['age', 'job', 'marital', 'education',
                'default', 'balance', 'housing', 'loan',
                'contact', 'day', 'month', 'duration',
                'campaign', 'pdays', 'previous', 'poutcome', 'y']

numeric_attr = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']

print("Loading in bank train data and test data")
train_data = pd.read_csv('bank_data/train.csv', names= bank_columns)
test_data = pd.read_csv('bank_data/test.csv', names = bank_columns)
print("Done")
print("----------------------------------")

#convert numeric to binary values:
def convert_numeric_to_binary(data, attrs):
    data= data.copy()
    for attr in attrs:
        median = data[attr].median()
        data[attr] = np.where(data[attr] <= median, f"<= {median}", f"> {median}")
    return data

gain_methods = ['information_gain', 'majority_error', 'gini']


#A dictionary so can turn to dataframe later.
results_a= {
    'Depth': [],
    'Method': [],
    'Train Error': [],
    'Test Error': []
}

results_b= {
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


#if doing 3b, instead of "train_data", fill in "fill_vals_train" as first param below this line"
revised_test_data = convert_numeric_to_binary(test_data, numeric_attr)

for i in range(2):
    if i==0:
        revised_train_data = convert_numeric_to_binary(train_data, numeric_attr)
        print("Begin training for question: 2.3.a, consider unknown as a value")
        print("----------------------------------")
    else:
        fill_vals_train = fill_missing_values(train_data, bank_columns)
        revised_train_data = convert_numeric_to_binary(fill_vals_train, numeric_attr)
        print("Begin training for question: 2.3.b, fill unknown value with majority value.")
        print("----------------------------------")

    for depth in range(1, 17):
        for method in gain_methods:

            print(f"Current depth: {depth}, training using: {method}")
            tree = decision_tree.DecisionTree(revised_train_data, 'y', attributes=bank_columns[:-1], gain_method=method, max_depth=depth)
            #predict
            train_predict = decision_tree.predict(tree, revised_train_data)
            test_predict = decision_tree.predict(tree, revised_test_data)

            # Calculate errors
            train_error = decision_tree.calculate_error(train_predict, revised_train_data['y'])
            test_error = decision_tree.calculate_error(test_predict, revised_test_data['y'])

            if (i==0):
                results_a['Depth'].append(depth)
                results_a['Method'].append(method)
                results_a['Train Error'].append(train_error)
                results_a['Test Error'].append(test_error)
            else:
                results_b['Depth'].append(depth)
                results_b['Method'].append(method)
                results_b['Train Error'].append(train_error)
                results_b['Test Error'].append(test_error)
        print("done depth: ", depth)
        print("----------------------------------")
    if i==0:
        print("---------Done 2.3.a, here's the result: ----------")
        results_df = pd.DataFrame(results_a)
        print(results_df)

    else:
        print("---------Done 2.3.b, here's the result: ----------")
        results_df = pd.DataFrame(results_b)
        print(results_df)
    print("----------------------------------")