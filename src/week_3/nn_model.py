# Created by @nyutal on 06/05/2020
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class NNModel:
    def __init__(self, n_h):
        self._n_h = n_h
        self._parameters = None
    
    def _init_params(self, x, y, seed=None):
        if seed:
            np.random.seed(seed)
        w_1 = np.random.randn(self._n_h, x.shape[0]) * 0.01
        b_1 = np.zeros((self._n_h, 1))
        w_2 = np.random.randn(y.shape[0], self._n_h) * 0.01
        b_2 = np.zeros((y.shape[0], 1))
        
        return w_1, b_1, w_2, b_2
    
    def _extract_parameters(self):
        assert self._parameters, 'must to initialize'
        w_1, b_1, w_2, b_2 = self._parameters
        return w_1, b_1, w_2, b_2
    
    def _save_parameters(self, w_1, b_1, w_2, b_2):
        self._parameters = w_1, b_1, w_2, b_2
    
    @staticmethod
    def _forward_propagation(parameters, x):
        w_1, b_1, w_2, b_2 = parameters
        z_1 = np.dot(w_1, x) + b_1
        a_1 = np.tanh(z_1)
        z_2 = np.dot(w_2, a_1) + b_2
        a_2 = sigmoid(z_2)
        assert (a_2.shape == (1, x.shape[1]))
        return z_1, a_1, z_2, a_2
    
    @staticmethod
    def _compute_cost(a_2, y):
        m = y.shape[1]
        cost = -(np.dot(np.log(a_2), y.T) + np.dot(np.log(1 - a_2), (1 - y).T)) / m
        return float(np.squeeze(cost))
    
    @staticmethod
    def _backward_propagation(parameters, a_1, a_2, x, y):
        m = x.shape[1]
        w_1, b_1, w_2, b_2 = parameters
        
        dz_2 = a_2 - y
        dw_2 = np.dot(dz_2, a_1.T) / m
        db_2 = np.sum(dz_2, axis=1, keepdims=True) / m
        d_z_1 = np.dot(w_2.T, dz_2) * (1 - np.power(a_1, 2))
        dw_1 = np.dot(d_z_1, x.T) / m
        db_1 = np.sum(d_z_1, axis=1, keepdims=True) / m
        
        return dw_1, db_1, dw_2, db_2
    
    @staticmethod
    def _update_parameters(parameters, grads, learning_rate=1.2):
        w_1, b_1, w_2, b_2 = parameters
        dw_1, db_1, dw_2, db_2 = grads
        w_1 = w_1 - learning_rate * dw_1
        b_1 = b_1 - learning_rate * db_1
        w_2 = w_2 - learning_rate * dw_2
        b_2 = b_2 - learning_rate * db_2
        return w_1, b_1, w_2, b_2
    
    def fit(self, x, y, num_iteration, learning_rate, verbose=True, seed=None):
        results = {
            'cost': []
        }
        parameters = self._init_params(x, y, seed)
        cost = []
        for i in range(num_iteration):
            z_1, a_1, z_2, a_2 = self._forward_propagation(parameters, x)
            cost = self._compute_cost(a_2, y)
            grads = self._backward_propagation(parameters, a_1, a_2, x, y)
            parameters = self._update_parameters(parameters, grads, learning_rate)
            if verbose and i % 1000 == 0:
                print(f'Cost after iteration {i}: {cost}')
            results['cost'].append(cost)
        self._save_parameters(*parameters)
        return results
    
    def predict(self, x):
        parameters = self._extract_parameters()
        z_1, a_1, z_2, a_2 = self._forward_propagation(parameters, x)
        return (a_2 > 0.5).astype(int)
