# coding: utf-8
# author: Junwei Deng

from legotsft import BaseProcessUnit

class DfInput(BaseProcessUnit):

    def __init__(self, 
                 time_col=None, 
                 target_col=None,
                 extra_feature_col=[],
                 **config):
        self.config = {"time_col": time_col,
                       "target_col": target_col,
                       "extra_feature_col": extra_feature_col}
        self.input_type = set(["dataframe"])
        self.output_type = set(["dataframe"])
        self._pre_processing = True
        self._post_processing = False
        self._input_layer = True
    
    def backward(self, x, **config):
        return x

    def forward(self, x, **config):
        if self.config["time_col"]:
            x = x.set_index(self.config["time_col"])
        
        if self.config["target_col"] is None:
            self.config["target_col"] = list(x.columns)

        if self.config["extra_feature_col"]:
            x = x[self.config["extra_feature_col"] + self.config["target_col"]]
        
        target_dim = []
        if self.config["target_col"]:
            for col in self.config["target_col"]:
                target_dim.append(list(x.columns).index(col))
        
        self.config["target_dim"] = target_dim

        return x