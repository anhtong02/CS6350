'''
This file is for loading in, testing data
Note: Decision Tree only works with data that is a pandas Dataframe with columns in it.
'''

import pandas as pd
import numpy as np

#load in the datasets for car

car_labels = ['unacc', 'acc', 'good', 'vgood']
car_attributes = ['buying', 'maint' ,'doors' ,'persons' ,'lug_boot' ,'safety' ,'label']

train_data = pd.read_csv('car_data/train.csv', names= car_attributes)
test_data = pd.read_csv('car_data/test.csv', header=None, names = car_attributes)


#print sumthing
values, counts = np.unique(train_data.iloc[:, 0], return_counts=True)
a=  [1, 2, 3,4]
print( sum(f**2 for f in a))