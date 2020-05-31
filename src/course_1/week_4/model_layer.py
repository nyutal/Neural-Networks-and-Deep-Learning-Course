# Created by @nyutal on 13/05/2020
from typing import List
from .grads import Grads
import numpy as np


class Layer:
    def __init__(self, to_cache=True):
        self._to_cache = to_cache
        self._cache = None
    
    def extract_params(self):
        return None
    
    def initialize_parameters(self, seed=None):
        if seed:
            np.random.seed(seed)
    
    def forward(self, input_data):
        pass
    
    def backward(self, output_tag):
        pass
    
    def update_parameters(self, learning_rate):
        pass


class Linear(Layer):
    def __init__(self, n_input, n_h, to_cache=True):
        super().__init__(to_cache)
        self._n_input = n_input
        self._n_h = n_h
        self._parameters = None
        self._grads = None
    
    def _dim(self):
        return self._n_input, self._n_h
    
    def _pack_params(self, w, b):
        self._parameters = (w, b)
    
    def extract_params(self):
        return self._parameters
    
    def initialize_parameters(self, seed=None):
        super().initialize_parameters(seed)
        n_x, n_h = self._dim()
        w = np.random.randn(self._n_h, n_x) * 0.01
        b = np.zeros((n_h, 1))
        self._pack_params(w, b)
    
    def forward(self, input_data):
        w, b = self.extract_params()
        assert input_data.shape[0] == w.shape[1]
        z = np.dot(w, input_data) + b
        if self._to_cache:
            self._cache = input_data
        return z
    
    def backward(self, d_output_data):
        assert self._cache is not None
        input_data = self._cache
        self._cache = None
        d_z = d_output_data
        m = input_data.shape[1]
        d_w = np.dot(d_z, input_data.T) / m
        d_b = np.sum(d_z, axis=1, keepdims=True) / m
        w, _ = self.extract_params()
        d_input_data = np.dot(w.T, d_z)
        self._grads = Grads((d_w, d_b), d_input_data)
        return self._grads.input()
    
    def update_parameters(self, learning_rate):
        assert bool(self._grads)
        w, b = self.extract_params()
        d_w, d_b = self._grads.inner()
        w = w - learning_rate * d_w
        b = b - learning_rate * d_b
        self._pack_params(w, b)


class Sigmoid(Layer):
    def backward(self, output_tag):
        assert self._cache is not None
        z = self._cache
        d_a = output_tag
        s = 1 / (1 + np.exp(-z))
        d_z = d_a * s * (1 - s)
        
        assert (d_z.shape == z.shape)
        return d_z
    
    def forward(self, input_data):
        a = 1 / (1 + np.exp(-input_data))
        if self._to_cache:
            self._cache = input_data
        return a


class Relu(Layer):
    def backward(self, output_tag):
        assert self._cache is not None
        z = self._cache
        d_a = output_tag
        d_z = np.array(d_a, copy=True)  # just converting d_z to a correct object.
        
        # When z <= 0, you should set dz to 0 as well.
        d_z[z <= 0] = 0
        
        assert (d_z.shape == z.shape)

        return d_z
    
    def forward(self, input_data):
        if self._to_cache:
            self._cache = input_data
        return np.maximum(0, input_data)


class Stacked(Layer):
    def __init__(self, layers: List[Layer], to_cache=True):
        super().__init__(to_cache)
        self._layers: List[Layer] = layers
    
    def extract_params(self):
        return [l.extract_params() for l in self._layers]
    
    def initialize_parameters(self, seed=None):
        super().initialize_parameters(seed)
        for l in self._layers:
            l.initialize_parameters()

    def backward(self, output_tag):
        d_input = output_tag
        for l in reversed(self._layers):
            d_input = l.backward(d_input)
        return d_input

    def forward(self, input_data):
        next_input = input_data
        for l in self._layers:
            next_input = l.forward(next_input)
        return next_input

    def update_parameters(self, learning_rate):
        for l in self._layers:
            l.update_parameters(learning_rate)
