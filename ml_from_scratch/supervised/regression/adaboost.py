import numpy as np 
import pandas as pd 
from random import sample
from math import log,exp
import random
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.metrics import confusion_matrix
# from sklearn import tree



dat=pd.read_csv("/home/danieldroder/Coding/machine_learning_explained/ml_from_scratch/supervised/regression/iris.csv")
example = dat[(dat['Species'] == 'versicolor') | (dat['Species'] == 'virginica')]
example['Label'] = example['Species'].replace(to_replace = ['versicolor','virginica'], value=[1,0])

class adaboost:
    def __init__(self):
        pass
    def _compute_error(self,y,y_pred,w_i):
        return (sum(w_i * (np.not_equal(y, y_pred)).astype(int)))/sum(w_i)
    def _compute_alpha(self,error):
        return np.log((1-error)/error)
    def _update_weights(w_i,alpha,y,y_pred):
        return w_i*np.exp(alpha*(np.not_equal(y,y_pred)).astype(int))



