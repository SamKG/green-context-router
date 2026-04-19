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

    num_iters = 50

    start_event1 = torch.cuda.Event(enable_timing=True)
    end_event1 = torch.cuda.Event(enable_timing=True)
    start_event2 = torch.cuda.Event(enable_timing=True)
    end_event2 = torch.cuda.Event(enable_timing=True)

    print(f"\nRunning Baseline (Single Stream, {num_iters} iterations)", flush=True)
    if "GREEN_CTX" in os.environ:
        del os.environ["GREEN_CTX"]
    
    torch.cuda.synchronize()
    start = time.perf_counter()
    
    start_event1.record()
    for _ in range(num_iters):
        torch.matmul(a, b)
    end_event1.record()
    
    torch.cuda.synchronize()
    print(f"Baseline Total time: {time.perf_counter() - start:.4f}s", flush=True)
    print(f"  Stream time: {start_event1.elapsed_time(end_event1):.2f} ms", flush=True)

    print(f"\nRunning with default context (No GREEN_CTX, {num_iters} iterations per stream)", flush=True)
    
    stream1 = torch.cuda.Stream()
    stream2 = torch.cuda.Stream()
    
    torch.cuda.synchronize()
    start = time.perf_counter()
    
    with torch.cuda.stream(stream1):
        start_event1.record(stream1)
        for _ in range(num_iters):
            torch.matmul(a, b)
        end_event1.record(stream1)
        
    with torch.cuda.stream(stream2):
        start_event2.record(stream2)
        for _ in range(num_iters):
            torch.matmul(a, b)
        end_event2.record(stream2)
        
    torch.cuda.synchronize()
    print(f"Default Total time: {time.perf_counter() - start:.4f}s", flush=True)
    print(f"  Stream 1 time: {start_event1.elapsed_time(end_event1):.2f} ms", flush=True)
    print(f"  Stream 2 time: {start_event2.elapsed_time(end_event2):.2f} ms", flush=True)

    print(f"\nRunning with disjoint Green Contexts (0 and 1, {num_iters} iterations per stream)", flush=True)
    
    # Create stream 1 with GREEN_CTX = 0 (e.g. 8 SMs)
    stream1_green = create_green_stream(0)
    
    # Create stream 2 with GREEN_CTX = 1 (e.g. remainder SMs)
    stream2_green = create_green_stream(1)
    
    torch.cuda.synchronize()
    start = time.perf_counter()
    
    with torch.cuda.stream(stream1_green):
        start_event1.record(stream1_green)
        for _ in range(num_iters):
            torch.matmul(a, b)
        end_event1.record(stream1_green)
            
    with torch.cuda.stream(stream2_green):
        start_event2.record(stream2_green)
        for _ in range(num_iters):
            torch.matmul(a, b)
        end_event2.record(stream2_green)
            
    torch.cuda.synchronize()
    print(f"Green Context Total time: {time.perf_counter() - start:.4f}s", flush=True)
    print(f"  Stream 1 (8 SMs) time: {start_event1.elapsed_time(end_event1):.2f} ms", flush=True)
    print(f"  Stream 2 (124 SMs) time: {start_event2.elapsed_time(end_event2):.2f} ms", flush=True)

if __name__ == "__main__":
    run_test()
