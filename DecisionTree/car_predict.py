'''
This file is for loading in, testing data
Note: Decision Tree only works with data that is a pandas Dataframe with columns in it.
'''

import pandas as pd

import decision_tree

#load in the datasets for car
car_labels = ['unacc', 'acc', 'good', 'vgood']
car_columns = ['buying', 'maint' ,'doors' ,'persons' ,'lug_boot' ,'safety' ,'label']

train_data = pd.read_csv('car_data/train.csv', names= car_columns)
test_data = pd.read_csv('car_data/test.csv', names = car_columns)


gain_methods = [ 'information_gain','majority_error', 'gini']


#A dictionary so can turn to dataframe later.
results = {
    'Depth': [],
    'Method': [],
    'Train Error': [],
    'Test Error': []
}


print("Question 2.2.b: Begin training for depth from 1 to 6")
print("----------------------------------")
for depth in range(1, 7):

    for method in gain_methods:
        # Train decision tree
        tree = decision_tree.DecisionTree(train_data, 'label', car_columns[:-1], gain_method=method, max_depth=depth)

        train_predict = decision_tree.predict(tree, train_data)
        train_error = decision_tree.calculate_error(train_predict, train_data['label'])

        test_predict = decision_tree.predict(tree, test_data)
        test_error = decision_tree.calculate_error(test_predict, test_data['label'])

        results['Depth'].append(depth)
        results['Method'].append(method)
        results['Train Error'].append(train_error)
        results['Test Error'].append(test_error)

complete_results_df = pd.DataFrame(results)
print("---------Done, here's the result for 2.2.b: -----------")

print(complete_results_df)
print("----------------------------------")
