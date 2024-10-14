'''
Decision tree using ID3 algorithm
'''

import numpy as np
from . import gain_methods as gm
import pandas as pd


def DecisionTree(data, label, attributes, gain_method = 'information_gain', max_depth= 1):

    """
    A decision tree, using ID3 algorithm. Using dictionary.
    :param data: Pandas DataFrame.
    :param label: a string, name of label column
    :param attributes: a list, contains all possible attributes in a DataFrame
    :param gain_method: a string, 3 possible methods: information_gain, majority_error, gini
    :param max_depth: int, allow max depth for user
    :return: a decision tree
    """

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
    root = {}

    #best split attribute
    if gain_method == 'information_gain':
        A = best_attribute(data, attributes, label,gm.information_gain)
    elif gain_method == 'majority_error':
        A = best_attribute(data, attributes, label, gm.majority_error_gain)
    elif gain_method == 'gini':
        A = best_attribute(data, attributes, label, gm.gini_gain)
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
    """
    Helper method for DecisionTree, helps the tree to pick the best attribute.
    :param data: Pandas DataFrame.
    :param attributes: A list.
    :param label: a string, name of label column.
    :param gain_method: one of the methods from gain_methods.py
    :return: name (type string) of the best attribute.
    """

    #Get Entropy(label) first
    label_column = data.iloc[:, -1]
    entropy = gm.entropy(label_column)

    information_gains = []
    for attr in attributes:
        attr_index_num = data.columns.get_loc(attr)
        info_gain = gain_method(data, attr_index_num, label)
        information_gains.append(info_gain)

    #return attribute that has the highest info gain
    index_of_best_attr = np.argmax(information_gains)
    return attributes[index_of_best_attr]


def predict(tree, data):
    """
    Method to predict using a trained decision tree
    :param tree: DecisionTree, a decision tree that's been trained
    :param data: Pandas DataFrame, a set of data used to predict, using tree.
    :return:
    """
    result = []
    for i, row in data.iterrows():
        node = tree

        #check if the node is dictionary, if yes then keep going.
        #if not then it is label, stop
        while isinstance(node, dict):

            attribute = list(node.keys())[0]
            current_val_of_attribute = row[attribute]

            #get the label of that value of an attribute
            node = node[attribute].get(current_val_of_attribute, None)

            #if there is not result then done
            if node is None:
                break
        result.append(node)
    return pd.Series(result)

# Function to calculate average prediction error
def calculate_error(predictions, true_labels):
    """
    Check the error between predictions and true label values.
    :param predictions: A Pandas Series
    :param true_labels: A Pandas Series
    :return:
    """

    return (predictions != true_labels).mean()