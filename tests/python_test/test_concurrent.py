import torch
import os
import time

def run_test():
    print("Warming up...")
    a = torch.randn(8192, 8192, device='cuda')
    b = torch.randn(8192, 8192, device='cuda')
    torch.matmul(a, b)
    torch.cuda.synchronize()

    stream1 = torch.cuda.Stream()
    stream2 = torch.cuda.Stream()

    print("Running with default context (No GREEN_CTX)")
    if "GREEN_CTX" in os.environ:
        del os.environ["GREEN_CTX"]
    
    start = time.perf_counter()
    with torch.cuda.stream(stream1):
        for _ in range(10):
            torch.matmul(a, b)
    with torch.cuda.stream(stream2):
        for _ in range(10):
            torch.matmul(a, b)
    torch.cuda.synchronize()
    print(f"Default time: {time.perf_counter() - start:.4f}s")

    print("Running with disjoint Green Contexts (0 and 1)")
    start = time.perf_counter()
    
    # Launch on stream 1 with GREEN_CTX = 0 (e.g. 8 SMs)
    os.environ["GREEN_CTX"] = "0"
    with torch.cuda.stream(stream1):
        for _ in range(10):
            torch.matmul(a, b)
            
    # Launch on stream 2 with GREEN_CTX = 1 (e.g. remainder SMs)
    os.environ["GREEN_CTX"] = "1"
    with torch.cuda.stream(stream2):
        for _ in range(10):
            torch.matmul(a, b)
            
    torch.cuda.synchronize()
    print(f"Green Context time: {time.perf_counter() - start:.4f}s")

if __name__ == "__main__":
    run_test()
