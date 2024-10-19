'''
Decision tree using ID3 algorithm
'''

import numpy as np
from . import gain_methods as gm
import pandas as pd
import joblib
from joblib import Parallel, delayed
import multiprocessing as mp

def DecisionTree(data, label, attributes, gain_method = 'information_gain', max_depth= None, weight = None, random_forest = None):
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

        most_common = data[label].value_counts().idxmax()  # Consider adding weights here

        if weight is not None:
            most_common = data.groupby(label).apply(lambda x: np.sum(x[weight])).idxmax()

        return most_common

    # 2) Otherwise:
    #create root node
    root = {}

    if random_forest:
        size = min(random_forest, len(attributes))
        selected_attributes = np.random.choice(attributes, size=size, replace=False)
    else:
        selected_attributes = attributes

    #best split attribute
    if gain_method == 'information_gain':
        A = best_attribute(data, selected_attributes, label,gm.information_gain, weight)
    elif gain_method == 'majority_error':
        A = best_attribute(data, selected_attributes, label, gm.majority_error_gain, weight)
    elif gain_method == 'gini':
        A = best_attribute(data, selected_attributes, label, gm.gini_gain, weight)
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
            Sv_attributes = [a for a in attributes if a != A]
            root[A][value] = DecisionTree(Sv, label, Sv_attributes, gain_method, max_depth-1 if max_depth is not None else None, weight)

    return root

def best_attribute(data, attributes, label, gain_method, weight = None):
    """
    Helper method for DecisionTree, helps the tree to pick the best attribute.
    :param data: Pandas DataFrame.
    :param attributes: A list.
    :param label: a string, name of label column.
    :param gain_method: one of the methods from gain_methods.py
    :return: name (type string) of the best attribute.
    """

    label_column = data.iloc[:, -1]

    information_gains = []
    for attr in attributes:
        attr_index_num = data.columns.get_loc(attr)


        info_gain = gain_method(data, attr_index_num, label, weight)
        information_gains.append(info_gain)

    #return attribute that has the highest info gain
    index_of_best_attr = np.argmax(information_gains)
    return attributes[index_of_best_attr]


def predict(tree, data, label):
    """
    Method to predict using a trained decision tree
    :param tree: DecisionTree, a decision tree that's been trained
    :param data: Pandas DataFrame, a set of data used to predict, using tree.
    :return: Pandas Series with predictions
    """
    result = []

    for i, row in data.iterrows():
        node = tree

        # Traverse the tree until reaching a leaf node (a label)
        while isinstance(node, dict):
            attribute = list(node.keys())[0]
            current_val_of_attribute = row[attribute]


            if ("<= " in current_val_of_attribute):
                threshold =  float(current_val_of_attribute.split("<= ")[1])

                vals_of_node_attr = node[attribute]

                for v in vals_of_node_attr:
                    if ("<= " in v):
                        val = float(v.split("<= ")[1])
                        if threshold <= val:
                                current_val_of_attribute = v
                    elif ("> " in v):
                        val = float(v.split("> ")[1])
                        if threshold > val:
                                current_val_of_attribute = v


            elif ("> " in current_val_of_attribute):
                threshold =  float(current_val_of_attribute.split("> ")[1])
                vals_of_node_attr = node[attribute]

                for v in vals_of_node_attr:
                    if ("<= " in v):
                        val = float(v.split("<= ")[1])
                        if threshold <= val:
                            current_val_of_attribute = v
                    elif ("> " in v):
                        val = float(v.split("> ")[1])
                        if threshold >= val:
                            current_val_of_attribute = v






            # Check if the attribute value exists in the tree
            if current_val_of_attribute in node[attribute]:
                node = node[attribute][current_val_of_attribute]
            else:
                subtree_values = [subtree for subtree in node[attribute].values() if isinstance(subtree, str)]

                if len(subtree_values) > 0:
                    node = pd.Series(subtree_values).mode()[0]
                else:
                    node = None
                break

        # If the node is None, fallback to most common label in the dataset
        if node is None:
            node = data[label].value_counts().idxmax()

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


