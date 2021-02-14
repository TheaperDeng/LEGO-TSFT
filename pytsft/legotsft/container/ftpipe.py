# coding: utf-8
# author: Junwei Deng

class FTpipe():

    def __init__(self, pu_list):
        self.pu_list = pu_list

    def forward(self, x):
        self.config = {}
        for pu in self.pu_list:
            x = pu.forward(x, **self.config)
            if pu._input_layer:
                self.config["target_dim"] = pu.config["target_dim"]
        return x
    
    def backward(self, x):
        for pu in self.pu_list[::-1]:
            x = pu.backward(x, **self.config)
        return x