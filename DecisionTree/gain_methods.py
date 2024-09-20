'''
Contains Info Gain methods: Entropy, GINI, Majority Error
'''
import numpy as np

# Information Gain (using Entropy)
def entropy(column):
    """
    Calculate entropy.
    :param column: Panda series
    :return: entropy
    """
    #get fraction of each values in the column
    fractions_of_vals = column.value_counts(normalize=True)
    entropy = 0
    for i in fractions_of_vals:
        entropy-= i * np.log2(i)
    return entropy

def information_gain(data, attr_index, label_column):

    """
    Information Gain method
    :param data: Pandas DataFrame
    :param attr_index: int, index of an attribute
    :param label_column: string, name of label column
    :return: info gain of an attribute.
    """

    total_entropy = entropy(data[label_column])
    # get all values and get num of examples in each value
    values, counts = np.unique(data.iloc[:, attr_index], return_counts=True)
    total = np.sum(counts)
    info_gain = 0

    for i in range(len(values)):
        #calculate entropy of each val
        extracted_vals = data[data.iloc[:, attr_index] == values[i]][label_column]
        val_entropy = entropy(extracted_vals)

        #calculate info gain:
        info_gain += (counts[i] / total) * val_entropy

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



    return 0

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