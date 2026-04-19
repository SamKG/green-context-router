import torch
import os
import time
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

def run_test():
    print("Warming up...", flush=True)
    a = torch.randn(8192, 8192, device='cuda')
    b = torch.randn(8192, 8192, device='cuda')
    torch.matmul(a, b)
    torch.cuda.synchronize()

    print("Running with default context (No GREEN_CTX)", flush=True)
    if "GREEN_CTX" in os.environ:
        del os.environ["GREEN_CTX"]
    
    stream1 = torch.cuda.Stream()
    stream2 = torch.cuda.Stream()
    
    start = time.perf_counter()
    with torch.cuda.stream(stream1):
        for _ in range(10):
            torch.matmul(a, b)
    with torch.cuda.stream(stream2):
        for _ in range(10):
            torch.matmul(a, b)
    torch.cuda.synchronize()
    print(f"Default Total time: {time.perf_counter() - start:.4f}s", flush=True)

    print("Running with disjoint Green Contexts (0 and 1)", flush=True)
    
    # Create stream 1 with GREEN_CTX = 0 (e.g. 8 SMs)
    stream1_green = create_green_stream(0)
    
    # Create stream 2 with GREEN_CTX = 1 (e.g. remainder SMs)
    stream2_green = create_green_stream(1)
    
    start = time.perf_counter()
    
    os.environ["GREEN_CTX"] = "0"
    with torch.cuda.stream(stream1_green):
        for _ in range(10):
            torch.matmul(a, b)
            
    os.environ["GREEN_CTX"] = "1"
    with torch.cuda.stream(stream2_green):
        for _ in range(10):
            torch.matmul(a, b)
            
    torch.cuda.synchronize()
    print(f"Green Context Total time: {time.perf_counter() - start:.4f}s", flush=True)

if __name__ == "__main__":
    run_test()
