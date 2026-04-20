import torch
import os
import time
import ctypes
from transformers import AutoModelForCausalLM

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
    model_id = "/scratch/gpfs/LI/samyakg/models/qwen/Qwen2.5-3B-Instruct/"
    print(f"Loading {model_id}...", flush=True)
    
    # We load in bfloat16 to fit easily in VRAM and compute fast, to 'cuda' directly
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.bfloat16, 
        device_map="cuda", 
        attn_implementation="eager"  # Use eager attention for easier stream execution analysis
    )
    
    # Mocking prefill inputs (Batch Size 1, Sequence Length 2048)
    seq_len = 128
    inputs1 = {"input_ids": torch.randint(0, 32000, (1, seq_len), device="cuda")}
    inputs2 = {"input_ids": torch.randint(0, 32000, (1, seq_len), device="cuda")}
    
    print("Warming up...", flush=True)
    with torch.no_grad():
        model(**inputs1)
    torch.cuda.synchronize()

    start_event1 = torch.cuda.Event(enable_timing=True)
    end_event1 = torch.cuda.Event(enable_timing=True)
    start_event2 = torch.cuda.Event(enable_timing=True)
    end_event2 = torch.cuda.Event(enable_timing=True)
    
    # ---------------------------------------------
    # Sequential Baseline
    # ---------------------------------------------
    print(f"\n--- Sequential Baseline (Seq Len {seq_len}) ---", flush=True)
    if "GREEN_CTX" in os.environ:
        del os.environ["GREEN_CTX"]
    torch.cuda.synchronize()
    start_seq = time.perf_counter()
    
    with torch.no_grad():
        start_event1.record()
        model(**inputs1)
        end_event1.record()
        
        start_event2.record()
        model(**inputs2)
        end_event2.record()
    
    torch.cuda.synchronize()
    print(f"Sequential Total time: {time.perf_counter() - start_seq:.4f}s", flush=True)
    print(f"  Stream 1 prefill time: {start_event1.elapsed_time(end_event1):.2f} ms", flush=True)
    print(f"  Stream 2 prefill time: {start_event2.elapsed_time(end_event2):.2f} ms", flush=True)

    # ---------------------------------------------
    # Default CUDA Streams
    # ---------------------------------------------
    print(f"\n--- Default CUDA Streams (Seq Len {seq_len}) ---", flush=True)
    stream1 = torch.cuda.Stream()
    stream2 = torch.cuda.Stream()
    
    torch.cuda.synchronize()
    start_def = time.perf_counter()
    
    with torch.no_grad():
        with torch.cuda.stream(stream1):
            start_event1.record(stream1)
            model(**inputs1)
            end_event1.record(stream1)
            
        with torch.cuda.stream(stream2):
            start_event2.record(stream2)
            model(**inputs2)
            end_event2.record(stream2)
            
    torch.cuda.synchronize()
    print(f"Default Streams Total time: {time.perf_counter() - start_def:.4f}s", flush=True)
    print(f"  Stream 1 prefill time: {start_event1.elapsed_time(end_event1):.2f} ms", flush=True)
    print(f"  Stream 2 prefill time: {start_event2.elapsed_time(end_event2):.2f} ms", flush=True)

    # ---------------------------------------------
    # Green Context Streams
    # ---------------------------------------------
    # We will iterate through different SM partitions like test_concurrent.py
    max_sms = 132 # Adjust based on your GPU
    num_pairs = (max_sms // 2) // 8

    print(f"\n--- Green Context Streams ---", flush=True)
    for i in range(num_pairs):
        ctx1_id = i * 2
        ctx2_id = i * 2 + 1
        sm1 = (i + 1) * 8
        sm2 = max_sms - sm1
        
        print(f"\nPartition: Context {ctx1_id} ({sm1} SMs) and Context {ctx2_id} ({sm2} SMs)", flush=True)
        
        stream1_green = create_green_stream(ctx1_id)
        stream2_green = create_green_stream(ctx2_id)
        
        torch.cuda.synchronize()
        start_green = time.perf_counter()
        
        with torch.no_grad():
            with torch.cuda.stream(stream1_green):
                start_event1.record(stream1_green)
                model(**inputs1)
                end_event1.record(stream1_green)
                    
            with torch.cuda.stream(stream2_green):
                start_event2.record(stream2_green)
                model(**inputs2)
                end_event2.record(stream2_green)
                    
        torch.cuda.synchronize()
        print(f"Green Context Streams Total time: {time.perf_counter() - start_green:.4f}s", flush=True)
        print(f"  Stream 1 ({sm1} SMs) prefill time: {start_event1.elapsed_time(end_event1):.2f} ms", flush=True)
        print(f"  Stream 2 ({sm2} SMs) prefill time: {start_event2.elapsed_time(end_event2):.2f} ms", flush=True)

if __name__ == "__main__":
    run_test()
