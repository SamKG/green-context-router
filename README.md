# Green Context Router

This project provides a dynamic router for CUDA Green Contexts using an `LD_PRELOAD` interposer. 

## 1. Goal of the Project
The primary goal is to provide a dynamic routing mechanism for CUDA kernel execution by allocating specific Streaming Multiprocessor (SM) partitions via CUDA Green Contexts (available in CUDA 12.4+). At startup (e.g., when `cuInit` is called or a context is required), the router creates a pool of all possible Green Contexts (in increments of 8) ranging from 8 SM to the device's maximum available SM count.

When downstream applications launch a kernel (by calling `cuLaunchKernel`, `cuLaunchKernelEx`, etc.), the router intercepts the call and reads the `GREEN_CTX` environment variable to determine which Green Context to swap in for the duration of the kernel launch.

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
export LD_PRELOAD=/path/to/green-ctx-router/target/release/libcuda.so.1
./your_cuda_application
```

## 4. Setting Environment Variables in Downstream Code

The `GREEN_CTX` environment variable determines the index of the Green Context pool (0-based) to use. Contexts are created in co-scheduled pairs. Index `0` corresponds to `8` SMs, index `1` corresponds to the remainder of the SMs (e.g., `124` SMs on a 132-SM GPU). Index `2` corresponds to `16` SMs, index `3` corresponds to the remainder of the SMs, and so on.

To use the router efficiently, downstream applications should dynamically set the `GREEN_CTX` environment variable immediately before the kernel launch. For instance, in PyTorch, you can do this before triggering a specific model operation:

```python
import os
import torch

# ... setup your model ...

# Route the next operations to a Green Context configured with 16 SMs (index 2)
os.environ["GREEN_CTX"] = "2"

# Perform the operation
output = model(input_tensor)

# Optionally unset or change it back for subsequent operations
os.environ["GREEN_CTX"] = "0"
```

## 5. Using the Debugging Environment Variable

The router utilizes the `tracing` framework for logging. By default, it is quiet, but you can enable informative output by setting the `GREEN_CTX_TRACE` environment variable.

- **For general info (like context mapping on each kernel launch):**
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
GREEN_CTX_TRACE=info LD_LIBRARY_PATH=/path/to/target/release:$LD_LIBRARY_PATH GREEN_CTX=4 ./your_cuda_app
```
r debugging:**
  ```bash
  export GREEN_CTX_TRACE=debug
  # or
  export GREEN_CTX_TRACE=trace
  ```

Combine it with your application execution like so:
```bash
GREEN_CTX_TRACE=info LD_PRELOAD=/path/to/target/release/libcuda.so.1 GREEN_CTX=4 ./your_cuda_app
```
