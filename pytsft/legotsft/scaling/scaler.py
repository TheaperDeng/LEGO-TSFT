# coding: utf-8
# author: Junwei Deng

from legotsft import BaseProcessUnit
from sklearn.preprocessing import StandardScaler
import numpy as np
import time
import pandas as pd

import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule


class StdScaler(BaseProcessUnit):
    def __init__(self, **config):
        super().__init__()
        self.input_type = set(["dataframe", "ndarray"])
        self.output_type = set(["ndarray"])
        self.internal = StandardScaler()
        self.config = {"mean": None, "std": None}
        self._pre_processing = True
        self._post_processing = True
        self._input_layer = False

    def _fit(self, x):
        self.internal.fit(x)
        self.config["mean"] = self.internal.mean_
        self.config["std"] = self.internal.scale_

    def _transform(self, x):
        return self.internal.transform(x)
    
    def forward(self, x, **config):
        self._fit(x)
        return self._transform(x)
    
    def backward(self, x, **config):
        target_col_indexes = config["target_dim"]
        x_dummy = np.zeros((x.shape[0], x.shape[1], self.config["std"].shape[0]))
        print(x_dummy[:, :, target_col_indexes].shape, x.shape)
        x_dummy[:, :, target_col_indexes] = x
        y_unscale = self.internal.inverse_transform(x_dummy)[:,:,target_col_indexes]
        return y_unscale

if __name__ == "__main__":
    x = np.random.rand(250*12, 200).astype(np.float32)
    # x = pd.DataFrame({"datetime": pd.date_range('1/1/2019', periods=50), 
    #                   "value1": np.random.randn(50),
    #                   "value2": np.random.randn(50)})
    # x = x.set_index("datetime")
    stdscaler = StdScaler()
    start_time = time.time()
    x_scaled = stdscaler.forward(x.copy())
    print("It takes {} seconds for cpu".format(time.time() - start_time))
    stack_x_scaled = np.stack([x_scaled, x_scaled], axis=0)
    pred = stdscaler.backward(stack_x_scaled)
    np.testing.assert_almost_equal(pred, np.stack([x,x], axis=0))
