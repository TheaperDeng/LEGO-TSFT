# coding: utf-8
# author: Junwei Deng

from legotsft import BaseProcessUnit
import numpy as np

class Df2Ndarray(BaseProcessUnit):
    def __init__(self, **config):
        self.input_type = set(["dataframe"])
        self.output_type = set(["ndarray"])
        self._pre_processing = True
        self._post_processing = False
        self._input_layer = False
        
    def forward(self, x, **config):
        x.reset_index(drop=True)
        x = np.stack([x[x.columns[i]].to_numpy() for i in range(len(x.columns))], axis=1)
        return x

    def backward(self, x, **config):
        return x