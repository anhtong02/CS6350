'''
Decision tree using ID3 algorithm
'''

import numpy as np
import gain_methods as gm
import pandas as pd


def DecisionTree(data, label, attributes, gain_method = 'information_gain', max_depth= None):
    labels = data[label].unique()

    #1) if all ex have same label, return that label:
    if len(labels) == 1:
        return labels[0]

    # if attributes empty or max depth is reached, return leaf node with the most common label
    if len(attributes) == 0 or (max_depth is not None and max_depth == 0):
        most_common = data[label].value_counts().idxmax() #get the most common label and return it
        return most_common

    # 2) Otherwise:
    #create root node
    root = None

    #best split attribute
    if gain_method == 'information_gain':
        A = best_attribute(data, attributes, gm.information_gain)
    elif gain_method == 'majority_error':
        A = best_attribute(data, attributes, gm.majority_error_gain)
    elif gain_method == 'gini':
        A = best_attribute(data, attributes, gm.gini_gain)
    else:
        raise ValueError("Choose one of these methods: 'information_gain', 'majority_error', or 'gini'.")

    root[A] = {}

    #for each possible vals of A can take:
    for value in data[A].unique():
        Sv = data[data[A] == value]

        if Sv.empty:
            root[A][value] = data[label].value_counts().idxmax()
        else:
            #get the rest of attributes:
            Sv_attributes = [a for a in attributes  if a != A]
            root[A][value] = DecisionTree(Sv, label, Sv_attributes, gain_method, max_depth-1 if max_depth is not None else None)

    return root

def best_attribute(data, attributes, label, gain_method):
    '''
    Method helps finding the best attribute. Can use Entropy, GINI, Majority Error
    '''

    #Get Entropy(label) first
    label_column = data.iloc[:, -1]
    entropy = gm.entropy(label_column)

    information_gains = []
    for attr in attributes:
        attr_index_num = data.columns.get_loc(attr)
        info_gain = gain_method(data, attr_index_num, label)
        information_gains.append(info_gain)

    #return attribute that has the highest info gain
    return attributes[np.argmax(information_gains)]