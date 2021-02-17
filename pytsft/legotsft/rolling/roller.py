# coding: utf-8
# author: Junwei Deng

from legotsft import BaseProcessUnit
import numpy as np
import pandas as pd

import time
import math

LIMIT_THREAD_PER_BLOCK = 800

class Roller(BaseProcessUnit):

    def __init__(self, lookback, horizon, **config):
        super().__init__()
        self.config = {"lookback": lookback,
                       "horizon": horizon,
                       "backend": "cpu"}
        self.config.update(config)
        self.input_type = set(["ndarray"])
        self.output_type = set(["ndarray"])
        self._pre_processing = True
        self._post_processing = False
        self._input_layer = False

        try:
            import pycuda.autoinit
            import pycuda.driver as cuda
            from pycuda.compiler import SourceModule
            mod = SourceModule("""
                __global__ void cuda_rolling(float *dest_x, float *dest_y, float *x, int lookback, int horizon, int length, int overset, int* target_dim, int target_dim_length, int step_x, int step_y)
                {
                    int block_idx = blockIdx.x;
                    int idx = threadIdx.x + block_idx*blockDim.x;
                    int idx_x = idx * lookback * overset;
                    int idx_y = idx * horizon * target_dim_length;
                    
                    if (idx+lookback+horizon-1 >= length){
                        return;
                    }

                    // dest_x
                    for (int i=0; i<=(lookback-1)*overset; i+=overset){
                        memcpy(&dest_x[idx_x+i], &x[idx*overset + i], sizeof(float)*overset);
                    }

                    // dest_y
                    for (int i=0; i<=(horizon-1); i+=1){
                        for (int j=0; j<target_dim_length; j++){
                            memcpy(&dest_y[idx_y+i*target_dim_length+j], &x[idx*overset+i*overset+lookback*overset+target_dim[j]], sizeof(float));
                        }
                    }
                }
            """)
            self._cuda_rolling_kernal = mod.get_function("cuda_rolling")
        except:
            print("You should install pycuda to use a cuda backend. You can still use cpu backend")
    
    def _rolling(self, x, target_dim=None):
        target_dim = target_dim if target_dim else list(range(0,x.shape[1]))
        out_x = np.stack([x.take(indices=range(i, i+self.config["lookback"]), axis=0)
            for i in range(x.shape[0] - self.config["lookback"] - self.config["horizon"] + 1)], axis=0)
        out_y = np.stack([x.take(indices=range(i+self.config["lookback"], i+self.config["lookback"] + self.config["horizon"]), axis=0)
            for i in range(x.shape[0] - self.config["lookback"] - self.config["horizon"] + 1)], axis=0)
        out_y = out_y.take(indices=target_dim, axis=2)
        return out_x, out_y

    def _cuda_rolling(self, x, target_dim=None):
        import pycuda.autoinit
        import pycuda.driver as cuda
        from pycuda.compiler import SourceModule
        target_dim = target_dim if target_dim else list(range(0,x.shape[1]))
        target_dim_length = len(target_dim)
        dest_x = np.zeros((x.shape[0]-self.config["lookback"]-self.config["horizon"]+1,self.config["lookback"]) + tuple([x.shape[1]])).astype(np.float32)
        dest_y = np.zeros((x.shape[0]-self.config["lookback"]-self.config["horizon"]+1,self.config["horizon"]) + tuple([target_dim_length])).astype(np.float32)
        target_dim = np.array(target_dim).astype(np.int32)
        step_x, step_y = x.strides
        step_x = step_x//4
        step_y = step_y//4
        self._cuda_rolling_kernal(
            cuda.Out(dest_x), 
            cuda.Out(dest_y), 
            cuda.In(x), 
            np.int32(self.config["lookback"]), 
            np.int32(self.config["horizon"]),
            np.int32(x.shape[0]),
            np.int32(np.prod(x.shape[1:])),
            cuda.In(target_dim),
            np.int32(target_dim_length),
            np.int32(step_x),
            np.int32(step_y),
            block=(LIMIT_THREAD_PER_BLOCK,1,1),
            grid=(math.ceil(x.shape[0]/LIMIT_THREAD_PER_BLOCK),1)
        )
        return dest_x, dest_y

    def forward(self, x, target_dim=None, **config):
        x = x.astype(np.float32)
        if self.config["backend"] == "cpu":
            return self._rolling(x, target_dim=target_dim)
        if self.config["backend"] == "cuda":
            try:
                self._cuda_rolling_kernal
            except:
                print("You should install pycuda to use a cuda backend. You can still use cpu backend")
                return
            return self._cuda_rolling(x, target_dim=target_dim)
    
    def backward(self, x, **config):
        return x

if __name__ == "__main__":
    x = pd.DataFrame({"value1": np.random.randn(5).astype(np.float32),
                      "value2": np.random.randn(5).astype(np.float32)})
    x = np.stack((x[x.columns[i]].to_numpy() for i in range(len(x.columns))), axis=1)
    roller = Roller(lookback=4, horizon=1, backend="cuda")
    start_time = time.time()
    out_x, out_y = roller.forward(x, target_dim=None)
    print("It takes {} seconds for gpu".format(time.time() - start_time))
    roller = Roller(lookback=4, horizon=1, backend="cpu")
    start_time = time.time()
    out_x_cpu, out_y_cpu = roller.forward(x, target_dim=None)
    print("It takes {} seconds for cpu".format(time.time() - start_time))
    np.testing.assert_almost_equal(out_x, out_x_cpu)
    np.testing.assert_almost_equal(out_y, out_y_cpu)

    x = np.random.rand(50, 2).astype(np.float32)
    roller = Roller(lookback=40, horizon=1, backend="cuda")
    start_time = time.time()
    out_x, out_y = roller.forward(x, target_dim=None)
    print("It takes {} seconds for gpu".format(time.time() - start_time))
    roller = Roller(lookback=40, horizon=1, backend="cpu")
    start_time = time.time()
    out_x_cpu, out_y_cpu = roller.forward(x, target_dim=None)
    print("It takes {} seconds for cpu".format(time.time() - start_time))
    np.testing.assert_almost_equal(out_x, out_x_cpu)
    np.testing.assert_almost_equal(out_y, out_y_cpu)

    
