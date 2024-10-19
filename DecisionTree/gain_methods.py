'''
Contains Info Gain methods: Entropy, GINI, Majority Error
Revised for HW2, decision tree must be able to take into account of weighted examples.
'''
import numpy as np
import pandas as pd




# Information Gain (using Entropy)
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

# Majority Error
def majority_error(column):

    majority_val = 1 - np.max(column.value_counts(normalize=True))
    return majority_val

def majority_error_gain(data, attr_index, label_column):

    total_ME = majority_error(data[label_column])
    # get all values and get num of examples in each value

    values, counts = np.unique(data.iloc[:, attr_index], return_counts=True)
    total = np.sum(counts)
    info_gain = 0

    for i in range(len(values)):
        # calculate ME of each val
        extracted_vals = data[data.iloc[:, attr_index] == values[i]][label_column]
        val_ME = majority_error(extracted_vals)

        # calculate info gain:
        info_gain += (counts[i] / total) * val_ME

    return total_ME - info_gain


# Gini Index
def gini_index(column):
    fractions_of_vals = column.value_counts(normalize=True)
    gini_val = 1 - sum(f**2 for f in fractions_of_vals)
    return gini_val

def gini_gain(data, attr_index, label_column):


    total_GI = gini_index(data[label_column])
    # get name of all values in an attribute and get num of examples in each value
    values, counts = np.unique(data.iloc[:, attr_index], return_counts=True)
    total = np.sum(counts) #total examples

    info_gain = 0
    for i in range(len(values)):
        # calculate ME of each val
        extracted_vals = data[data.iloc[:, attr_index] == values[i]][label_column]
        val_GI = gini_index(extracted_vals)

        # calculate info gain:
        info_gain += (counts[i] / total) * val_GI

    return total_GI - info_gain



