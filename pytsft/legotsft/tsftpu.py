# coding: utf-8
# author: Junwei Deng

class BaseProcessUnit:
    
    def __init__(self):
        self.input_type = None
        self.output_type = None
        self.config = None
    
    def forward(self, x):
        pass