# CS6350 
This is a machine learning library developed by Anh Tong for
CS5350/6350 in University of Utah

## How to use decision tree:
The algorithm is in decision_tree.py, and the method is : DecisionTree()

1. Call it: `decision_tree.DecisionTree()` , check out next step for parameter.
2. Parameter: `def DecisionTree(data, label, attributes, gain_method = 'information_gain', max_depth= None, weight = None, random_forest = None)`
   
`data` : training data, a pandas dataframe with **headers**

`label` : a string indicates the column name for the dataset, for example in the bank dataset, it is 'y', and in car dataset, it's 'label'.

`attributes`: should be **a list**. Includes all the column names **except the label column**. For example in car dataset, it's = ['buying', 'maint' ,'doors' ,'persons' ,'lug_boot' ,'safety'], in bank, it's = ['age', 'job', 'marital', 'education',
                'default', 'balance', 'housing', 'loan',
                'contact', 'day', 'month', 'duration',
                'campaign', 'pdays', 'previous', 'poutcome']

`max_depth` : an int, your desire of depth for tree, if you dont want to set depth limit, dont include `max_depth` in your method.

`weight`: this is made specifically for AdaBoost. It is a string indicates name of column that contains weight in a dataframe. Check out line 34 and 41 in AdaBoost.py. If you dont need weight, ignore it.

`random_forest`: this is for random forest (hw2, question 2d), it takes an int indicate the subset of how many attributes you want, like 2 or 4 or 6.
