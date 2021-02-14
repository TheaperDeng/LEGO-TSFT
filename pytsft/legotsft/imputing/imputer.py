# coding: utf-8
# author: Junwei Deng

from legotsft import BaseProcessUnit

class FfillImputer(BaseProcessUnit):

    def __init__(self, **config):
        self.input_type = set(["dataframe"])
        self.output_type = set(["dataframe"])
        self._pre_processing = True
        self._post_processing = False
        self._input_layer = False
    
    def forward(self, x, **config):
        x.iloc[0] = x.iloc[0].fillna(0)
        return x.fillna(method='ffill')
    
    def backward(self, x, **config):
        return x

    