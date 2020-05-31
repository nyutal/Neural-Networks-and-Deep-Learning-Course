# Created by @nyutal on 13/05/2020
import numpy as np


class Cost:
    def __init__(self):
        pass
    
    def cost(self, y, y_hat):
        pass
    
    def cost_derivative(self, y, y_hat):
        pass


class CrossEntropy(Cost):
    def cost(self, y, y_hat):
        m = y.shape[1]
        cost = -(np.dot(y, np.log(y_hat.T)) + np.dot((1 - y), np.log((1 - y_hat).T))) / m
        
        cost = np.squeeze(cost)  # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
        assert (cost.shape == ())
        return cost
    
    def cost_derivative(self, y, y_hat):
        return -(np.divide(y, y_hat) - np.divide(1 - y, 1 - y_hat))
