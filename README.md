# Green Context Router

This project provides a dynamic router for CUDA Green Contexts using an `LD_PRELOAD` interposer. 

## 1. Goal of the Project
The primary goal is to provide a dynamic routing mechanism for CUDA kernel execution by allocating specific Streaming Multiprocessor (SM) partitions via CUDA Green Contexts (available in CUDA 12.4+). At startup (e.g., when `cuInit` is called or a context is required), the router creates a pool of all possible Green Contexts (in increments of 8) ranging from 8 SMs up to the device's maximum available SM count.

When downstream applications create a stream (by calling `cuStreamCreate` or `cuStreamCreateWithPriority`), the router intercepts the call and reads the `GREEN_CTX` environment variable to determine which Green Context to bind the stream to. Any kernel subsequently launched on this stream will be physically restricted to that Green Context's hardware partition.

## 2. Building for Code and Tests

A `Makefile` is included to simplify building the interposer library and the test binaries.

- **Build Everything:**
  ```bash
  make all
  ```

- **Build Only the Library Hooks:**
  ```bash
  make hooks
  ```

- **Build Only the Tests:**
  ```bash
  make tests
  ```

## 3. How to Run the Hook

To run an application with the Green Context Router, set the `LD_PRELOAD` environment variable to point to the compiled shared object:

```bash
export LD_PRELOAD=/path/to/green-ctx-router/target/release/libgreen_ctx_router.so:$LD_PRELOAD
./your_cuda_application
```

## 4. Setting Environment Variables in Downstream Code

The `GREEN_CTX` environment variable determines the index of the Green Context pool (0-based) to use. Contexts are created in co-scheduled pairs. Index `0` corresponds to `8` SMs, index `1` corresponds to the remainder of the SMs (e.g., `124` SMs on a 132-SM GPU). Index `2` corresponds to `16` SMs, index `3` corresponds to the remainder of the SMs, and so on.

To use the router efficiently, downstream applications should dynamically set the `GREEN_CTX` environment variable immediately before stream creation. For instance, in PyTorch, you can use `ctypes` to intercept the native C-call and bind the PyTorch stream to the router:

```python
import os
import torch
import ctypes

cudart = ctypes.CDLL("libcudart.so")
cudaStreamCreate = cudart.cudaStreamCreate
cudaStreamCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
cudaStreamCreate.restype = ctypes.c_int

def create_green_stream(green_ctx_id):
    os.environ["GREEN_CTX"] = str(green_ctx_id)
    stream_ptr = ctypes.c_void_p()
    res = cudaStreamCreate(ctypes.byref(stream_ptr))
    if res != 0:
        raise RuntimeError(f"cudaStreamCreate failed with error {res}")
    return torch.cuda.ExternalStream(stream_ptr.value)

# ... setup your model ...

# Create a stream explicitly bound to exactly 16 SMs (index 2)
stream16 = create_green_stream(2)

# Perform operations on this stream, naturally restricting execution to 16 SMs
with torch.cuda.stream(stream16):
    output = model(input_tensor)
```

## 5. Using the Debugging Environment Variable

The router utilizes the `tracing` framework for logging. By default, it is quiet, but you can enable informative output by setting the `GREEN_CTX_TRACE` environment variable.

- **For general info (like context mapping on stream creation):**
  ```bash
  export GREEN_CTX_TRACE=info
  ```

- **For detailed initialization or deeper debugging:**
  ```bash
  export GREEN_CTX_TRACE=debug
  # or
  export GREEN_CTX_TRACE=trace
  ```

Combine it with your application execution like so:
```bash
GREEN_CTX_TRACE=info LD_PRELOAD=/path/to/target/release/libgreen_ctx_router.so GREEN_CTX=4 ./your_cuda_app
```
