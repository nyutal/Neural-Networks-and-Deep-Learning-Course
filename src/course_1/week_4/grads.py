# Created by @nyutal on 13/05/2020


class Grads:
    def __init__(self, inner_grads, input_grads):
        self._inner = inner_grads
        self._input = input_grads
    
    def input(self):
        return self._input
    
    def inner(self):
        return self._inner
    
    def __repr__(self):
        return f'Grads(\ninner: {self.inner()}\ninput: {self.input()}'
