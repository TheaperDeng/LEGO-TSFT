# coding: utf-8
# author: Junwei Deng

from legotsft import BaseProcessUnit
import numpy as np

import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

import time
import math

LIMIT_THREAD_PER_BLOCK = 800

class Roller(BaseProcessUnit):

    def __init__(self, lookback, horizon, **config):
        super().__init__()
        self.config = {"lookback": lookback,
                       "horizon": horizon}
        self.input_type = set(["dataframe", "ndarray"])
        self.output_type = set(["ndarray"])
        mod = SourceModule("""
            __global__ void cuda_rolling(float *dest_x, float *dest_y, float *x, int lookback, int horizon, int length, int overset)
            {
                int block_idx = blockIdx.x;
                int idx = threadIdx.x + block_idx*blockDim.x;
                int idx_x = idx * lookback * overset;
                int idx_y = idx * horizon * overset;
                
                if (idx+lookback+horizon-1 >= length){
                    return;
                }

                // dest_x
                for (int i=0; i<=(lookback-1)*overset; i+=overset){
                    memcpy(&dest_x[idx_x+i], &x[idx*overset + i], sizeof(float)*overset);
                }

                // dest_y
                for (int i=0; i<=(horizon-1)*overset; i+=overset){
                    memcpy(&dest_y[idx_y+i], &x[idx*overset+i+lookback*overset], sizeof(float)*overset);
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
        dest_x = np.zeros((x.shape[0]-self.config["lookback"]-self.config["horizon"]+1,self.config["lookback"]) + x.shape[1:]).astype(np.float32)
        dest_y = np.zeros((x.shape[0]-self.config["lookback"]-self.config["horizon"]+1,self.config["horizon"]) + x.shape[1:]).astype(np.float32)
        self._cuda_rolling_kernal(
            cuda.Out(dest_x), 
            cuda.Out(dest_y), 
            cuda.In(x), 
            np.int32(self.config["lookback"]), 
            np.int32(self.config["horizon"]),
            np.int32(x.shape[0]),
            np.int32(np.prod(x.shape[1:])),
            block=(LIMIT_THREAD_PER_BLOCK,1,1),
            grid=(math.ceil(x.shape[0]/LIMIT_THREAD_PER_BLOCK),1)
        )
        return dest_x, dest_y

    def forward(self, x, backend="cpu", axis=0):
        if backend == "cpu":
            return self._rolling(x, axis=axis)
        if backend == "cuda":
            return self._cuda_rolling(x)

if __name__ == "__main__":
    x = np.random.rand(1000).astype(np.float32)
    roller = Roller(lookback=320, horizon=40)
    start_time = time.time()
    out_x, out_y = roller.forward(x, backend="cuda")
    print("It takes {} seconds for gpu".format(time.time() - start_time))
    start_time = time.time()
    out_x_cpu, out_y_cpu = roller.forward(x, backend="cpu")
    print("It takes {} seconds for cpu".format(time.time() - start_time))
    np.testing.assert_almost_equal(out_x, out_x_cpu)
    np.testing.assert_almost_equal(out_y, out_y_cpu)