import time

import torch


def measure_performance(model, data, name="Model", warmup=5, iters=20):
    device = data.device
    model = model.to(device)

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps":
            torch.mps.synchronize()

    def reset_mem():
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        # MPS does not support resetting peak memory stats currently

    def get_mem():
        if device.type == "cuda":
            return torch.cuda.max_memory_allocated() / (1024**2)  # MB
        elif device.type == "mps":
            return torch.mps.current_allocated_memory() / (1024**2)  # MB
        return 0

    try:
        # Warmup
        for _ in range(warmup):
            _ = model(data)
        sync()

        reset_mem()

        start_event = time.perf_counter()
        for _ in range(iters):
            _ = model(data)
        sync()
        end_event = time.perf_counter()

        avg_time = ((end_event - start_event) / iters) * 1000  # ms
        peak_mem = get_mem()
        return {"latency_ms": avg_time, "peak_mem_mb": peak_mem}
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"[{name}] OOM Error on {device}")
            return {"latency_ms": float("inf"), "peak_mem_mb": float("inf")}
        else:
            raise e
