import torch
import os
import time
import ctypes
import multiprocessing as mp
from transformers import AutoModelForCausalLM

def worker(proc_id, seq_len, command_queue, result_queue, barrier):
    model_id = "/scratch/gpfs/LI/samyakg/models/qwen/Qwen2.5-3B-Instruct/"
    # Load model once per process
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.bfloat16, 
        device_map="cuda", 
        attn_implementation="eager"
    )
    
    inputs = {"input_ids": torch.randint(0, 32000, (1, seq_len), device="cuda")}
    
    # Warmup
    print(f"[Process {proc_id}] Warming up...", flush=True)
    with torch.no_grad():
        model(**inputs)
    torch.cuda.synchronize()
    
    cudart = ctypes.CDLL("libcudart.so")
    cudaStreamCreate = cudart.cudaStreamCreate
    cudaStreamCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    cudaStreamCreate.restype = ctypes.c_int
    
    while True:
        cmd = command_queue.get()
        if cmd is None:
            break
            
        green_ctx_id = cmd.get("ctx_id")
        
        if green_ctx_id is not None:
            os.environ["GREEN_CTX"] = str(green_ctx_id)
            stream_ptr = ctypes.c_void_p()
            res = cudaStreamCreate(ctypes.byref(stream_ptr))
            if res != 0:
                raise RuntimeError(f"cudaStreamCreate failed with error {res}")
            stream = torch.cuda.ExternalStream(stream_ptr.value)
        else:
            if "GREEN_CTX" in os.environ:
                del os.environ["GREEN_CTX"]
            stream = torch.cuda.Stream()
            
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        # Wait for both processes to be ready
        barrier.wait()
        
        # Launch inference
        start_time = time.perf_counter()
        with torch.no_grad():
            with torch.cuda.stream(stream):
                start_event.record(stream)
                model(**inputs)
                end_event.record(stream)
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        
        wall_time = end_time - start_time
        gpu_time = start_event.elapsed_time(end_event)
        
        result_queue.put({
            "proc_id": proc_id,
            "wall_time": wall_time,
            "gpu_time": gpu_time
        })

def run_test():
    max_sms = 132
    num_pairs = (max_sms // 2) // 8
    seq_len = 128
    
    ctx = mp.get_context('fork')
    barrier = ctx.Barrier(2)
    cmd_q1 = ctx.Queue()
    cmd_q2 = ctx.Queue()
    res_q = ctx.Queue()
    
    print("Spawning processes and loading models (this takes a moment)...", flush=True)
    p1 = ctx.Process(target=worker, args=(1, seq_len, cmd_q1, res_q, barrier))
    p2 = ctx.Process(target=worker, args=(2, seq_len, cmd_q2, res_q, barrier))
    
    p1.start()
    p2.start()
    
    def execute_pair(ctx1, ctx2, label1, label2, num_runs=5):
        if not p1.is_alive() or not p2.is_alive():
            print("Workers are dead, skipping...")
            return
            
        total_wall1, total_gpu1 = 0, 0
        total_wall2, total_gpu2 = 0, 0
        total_max_wall = 0
        
        for _ in range(num_runs):
            cmd_q1.put({"ctx_id": ctx1})
            cmd_q2.put({"ctx_id": ctx2})
            
            try:
                import queue
                res1 = res_q.get(timeout=60)
                res2 = res_q.get(timeout=60)
            except queue.Empty:
                print("Timeout waiting for workers. They likely crashed.")
                return
                
            results = {res1['proc_id']: res1, res2['proc_id']: res2}
            total_wall1 += results[1]['wall_time']
            total_gpu1 += results[1]['gpu_time']
            total_wall2 += results[2]['wall_time']
            total_gpu2 += results[2]['gpu_time']
            total_max_wall += max(results[1]['wall_time'], results[2]['wall_time'])
            
        print(f"Partition: {label1} and {label2} (Avg over {num_runs} runs)", flush=True)
        print(f"  Total Max Wall Time: {total_max_wall / num_runs:.4f}s", flush=True)
        print(f"  Proc 1 Wall: {total_wall1 / num_runs:.4f}s, GPU: {total_gpu1 / num_runs:.2f}ms", flush=True)
        print(f"  Proc 2 Wall: {total_wall2 / num_runs:.4f}s, GPU: {total_gpu2 / num_runs:.2f}ms\n", flush=True)

    # Let the warmups finish
    time.sleep(10)
    
    print(f"\n--- Default CUDA Streams (No Partitioning) ---", flush=True)
    execute_pair(None, None, "Default", "Default")
    
    print(f"\n--- Green Context Streams ---", flush=True)
    for i in range(num_pairs):
        ctx1_id = i * 2
        ctx2_id = i * 2 + 1
        sm1 = (i + 1) * 8
        sm2 = max_sms - sm1
        execute_pair(ctx1_id, ctx2_id, f"Context {ctx1_id} ({sm1} SMs)", f"Context {ctx2_id} ({sm2} SMs)")
        
    cmd_q1.put(None)
    cmd_q2.put(None)
    
    p1.join()
    p2.join()

if __name__ == "__main__":
    run_test()
