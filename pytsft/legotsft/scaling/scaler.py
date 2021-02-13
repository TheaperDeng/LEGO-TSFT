# coding: utf-8
# author: Junwei Deng

from legotsft import BaseProcessUnit
from sklearn.preprocessing import StandardScaler
import numpy as np
import time

import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule


class StdScaler(BaseProcessUnit):
    def __init__(self, **config):
        super().__init__()
        self.input_type = set(["ndarray"])
        self.output_type = set(["ndarray"])
        self.internal = StandardScaler()
        self.config = {"mean": None, "std": None}

    def _fit(self, x):
        self.internal.fit(x)
        self.config["mean"] = self.internal.mean_
        self.config["std"] = self.internal.scale_

    def _transform(self, x):
        return self.internal.transform(x)
    
    def forward(self, x):
        self._fit(x)
        return self._transform(x)

if __name__ == "__main__":
    x = np.random.rand(250*12, 200).astype(np.float32)
    stdscaler = StdScaler()
    start_time = time.time()
    x_scaled = stdscaler.forward(x)
    print("It takes {} seconds for cpu".format(time.time() - start_time))
