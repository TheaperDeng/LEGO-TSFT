# coding: utf-8
# author: Junwei Deng

from legotsft import BaseProcessUnit

class NdarrayInput(BaseProcessUnit):
    def __init__(self, time_axis=0, target_dim=None, **config):
        self.input_type = set(["ndarray"])
        self.output_type = set(["ndarray"])
        self.config = {"time_axis": time_axis,
                       "target_dim": target_dim}
        self._pre_processing = True
        self._post_processing = False
        self._input_layer = True

    def forward(self, x, **config):
        assert x.ndim > 2, "We only support time series numpy array less than 2 dim"
        if self.config["time_axis"] != 0:
            x = x.transpose((1,0))
        return x
    
    def backward(self, x, **config):
        if self.config["time_axis"] != 0:
            if x.ndim == 3:
                x = x.transpose((2,1))
            else:
                x = x.transpose((1,0))
        return x