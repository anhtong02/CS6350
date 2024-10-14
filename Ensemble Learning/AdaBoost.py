from DecisionTree import decision_tree
import pandas as pd
import numpy as np
import math
car_labels = ['unacc', 'acc', 'good', 'vgood']
car_columns = ['buying', 'maint' ,'doors' ,'persons' ,'lug_boot' ,'safety' ,'label']


train_data = pd.read_csv('../DecisionTree/car_data/train.csv', names=car_columns)
train_data = pd.read_csv('C:/Users/Xiu/PycharmProjects/CS6350-HW1/DecisionTree/car_data/test.csv', names=car_columns)

class AdaBoost:
    def __init__(self, T=20):
        self.T = T
        self.alphas = []
        self.trees = []
        self.weights = None

    def Boosting(self, data, label, attributes):
        # initialize the weight:
        examples_size = len(data)
        init_weight = 1 / examples_size
        weights = np.full(examples_size, init_weight)

        data = data.copy()
        data['weights'] = weights

        for t in range(self.T):

            # 1)find a classifier ht whose weighted classification error is better than chance
            tree = decision_tree.DecisionTree(data, 'label', car_columns[:-1])
            self.trees.append(tree)

            # check if weighted classification error is better than chance
            # if e_T < , > , do...
            train_predict = decision_tree.predict(tree, data)

            #check if predictions and real data are same length
            if (len(train_predict)) !=len(data[label]):
                raise ValueError(f"length or predicted labels is not the same as "
                                 f"length of original labels column")

            #calculate error
            e_t = decision_tree.calculate_error(train_predict, data[label])
            if e_t <= 0 or e_t >= 1:
                print(f"Error {e_t} is not valid; skipping iteration {t}.")
                continue

            # 2) Compute alpha: a_t = 1/2ln( (1-e_t) / e_t )

            alpha_t = 1/2 * math.log( (1 - e_t) / e_t)
            self.alphas.append(alpha_t)


            # 3) update values of the weights for the each training examples
            """
            Dt+1(i) = Dt(i) / Zt * exp(-alpha_t * yiht(xi))
            
            e^-at if yi=ht(xi)
            e^at if yi != ht(xi)
            
            Zt = sum of all updated weighted examples at t.
            """
            for index, (true_label, prediction, weight) in enumerate(zip(data[label], train_predict, data['weights'])):
                # print(true_label, prediction, weight)
                yi_ht = -1 if true_label != prediction else 1

                #Dt+1
                next_weights = weight * math.exp(-alpha_t * yi_ht)

                #update the next weight in the data frame!
                data.at[index, 'weights'] = next_weights


            Zt = data['weights'].sum()

            #divide the weights by Zt
            data['weights'] = data['weights'] / Zt


    def predict(self, data):
        final_prediction = np.zeros(len(data))

        for alpha, tree in zip(self.alphas, self.trees):
            tree_predictions = decision_tree.predict(tree, data)
            final_prediction += alpha * tree_predictions

        return np.sign(final_prediction)


adaboost = AdaBoost()
a = adaboost.Boosting(train_data, 'label', car_columns)
