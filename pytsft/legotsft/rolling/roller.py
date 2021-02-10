# coding: utf-8
# author: Junwei Deng

from legotsft import BaseProcessUnit
import numpy as np

import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

import time

class Roller(BaseProcessUnit):

    def __init__(self, lookback, horizon, **config):
        super().__init__()
        self.config = {"lookback": lookback,
                       "horizon": horizon}
        self.input_type = set(["dataframe", "ndarray"])
        self.output_type = set(["ndarray"])
        mod = SourceModule("""
            __global__ void cuda_rolling(float *dest_x, float *dest_y, float *x, int lookback, int horizon)
            {
            int idx = threadIdx.x;
            int idx_x = idx * lookback;
            int idx_y = idx * horizon;
            for (int i=0; i<=lookback-1; i++){
                dest_x[idx_x+i] = x[threadIdx.x+i];
            }
            for (int i=0; i<=horizon-1; i++){
                dest_y[idx_y+i] = x[threadIdx.x+i+lookback];
            }
            }
        """)
        self._cuda_rolling_kernal = mod.get_function("cuda_rolling")
    
    def _rolling(self, x, axis=0):
        out_x = np.stack([x.take(indices=range(i, i+self.config["lookback"]), axis=axis)
            for i in range(x.shape[axis] - self.config["lookback"] - self.config["horizon"] + 1)], axis=0)
        out_y = np.stack([x.take(indices=range(i+self.config["lookback"], i+self.config["lookback"] + self.config["horizon"]), axis=axis)
            for i in range(x.shape[axis] - self.config["lookback"] - self.config["horizon"] + 1)], axis=0)
        return out_x, out_y

    def _cuda_rolling(self, x, axis=0):
        dest_x = np.zeros((x.shape[0]-self.config["lookback"]-self.config["horizon"]+1,self.config["lookback"])).astype(np.float32)
        dest_y = np.zeros((x.shape[0]-self.config["lookback"]-self.config["horizon"]+1,self.config["horizon"])).astype(np.float32)
        self._cuda_rolling_kernal(
            cuda.Out(dest_x), 
            cuda.Out(dest_y), 
            cuda.In(x), 
            np.int32(self.config["lookback"]), 
            np.int32(self.config["horizon"]),
            block=(x.shape[0],1,1), grid=(1,1)
        )
        return dest_x, dest_y


    def forward(self, x, backend="cpu"):
        if backend == "cpu":
            return self._rolling(x)
        if backend == "cuda":
            return self._cuda_rolling(x)

if __name__ == "__main__":
    x = np.arange(1000).astype(np.float32)
    roller = Roller(lookback=5, horizon=1)
    start_time = time.time()
    out_x, out_y = roller.forward(x, backend="cuda")
    print("It takes {} seconds for gpu".format(time.time() - start_time))
    print(out_x)
    start_time = time.time()
    out_x, out_y = roller.forward(x, backend="cpu")
    print("It takes {} seconds for cpu".format(time.time() - start_time))
    print(out_x)