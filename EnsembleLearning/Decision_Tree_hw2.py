'''
Decision tree using ID3 algorithm
'''

import numpy as np
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
        A = best_attribute(data, selected_attributes, label, information_gain, weight)

    else:
        raise ValueError("Right now there is not an update for taking weight into account using 'majority_error', or 'gini'."
                         "Please choose 'information_gain'")

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


        info_gain = information_gain(data, attr_index_num, label, weight)
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

def entropy(column, weight=None):
    """
    Calculate entropy, accounting for sample weights if provided.
    :param column: Pandas Series of labels.
    :param weight: Optional; weights for each sample.
    :return: entropy value (float).
    """
    entropy = 0

    if weight is not None:
        data = pd.DataFrame({'label': column, 'weights': weight})

        # Sum of weights for each label
        weighted_sums = data.groupby('label')['weights'].sum()

        # Normalize the weights
        total_weight = weight.sum()

        fractions_of_vals = weighted_sums / total_weight
    else:
        # Get fraction of each value in the column (unweighted case)
        fractions_of_vals = column.value_counts(normalize=True)

    # Calculate entropy
    for fraction in fractions_of_vals:
        if fraction > 0:
            entropy -= fraction * np.log2(fraction)

    return entropy


def information_gain(data, attr_index, label_column, weight=None):
    """
    Information Gain method considering sample weights if provided.
    :param data: Pandas DataFrame.
    :param attr_index: int, index of an attribute.
    :param label_column: string, name of label column.
    :param weight: Optional; weights for each sample.
    :return: information gain of an attribute.
    """
    # Calculate total entropy (either weighted or unweighted)
    if weight is not None:
        total_entropy = entropy(data[label_column], data[weight])
    else:
        total_entropy = entropy(data[label_column])

    # Get unique values for the attribute and their counts
    values = data.iloc[:, attr_index].unique()

    info_gain = 0

    # Total weight or count of examples
    if weight is not None:
        total_weight = data[weight].sum()
    else:
        total_weight = len(data)

    # Loop over all possible values of the attribute
    for value in values:
        subset = data[data.iloc[:, attr_index] == value]  # Subset data where attr == value

        # Get the corresponding labels in the subset
        extracted_vals = subset[label_column]

        if weight is not None:
            # Get weights of the subset
            extracted_weights = subset[weight]

            # Calculate entropy of the subset with weights
            val_entropy = entropy(extracted_vals, extracted_weights)

            # Scale by the fraction of total weight that this subset represents
            subset_weight_fraction = extracted_weights.sum() / total_weight

            # Weighted sum of entropy
            info_gain += subset_weight_fraction * val_entropy

        else:
            # Unweighted case
            val_entropy = entropy(extracted_vals)

            # Scale by fraction of total examples
            info_gain += (len(subset) / total_weight) * val_entropy

    return total_entropy - info_gain
