"""Sample GPU memory only inside generation windows; no model loading here."""
import threading
import time


def summarize(samples):
    if len(samples) < 2:
        raise ValueError("Memory measurement needs start and end samples")
    duration = samples[-1]["seconds"] - samples[0]["seconds"]
    if duration <= 0:
        raise ValueError("Non-positive measurement duration")
    result = {"duration_seconds": duration, "sample_count": len(samples)}
    for key in samples[0]:
        if key == "seconds":
            continue
        area = sum((b["seconds"]-a["seconds"])*(a[key]+b[key])/2
                   for a,b in zip(samples,samples[1:]))
        result[key+"_mean"] = area/duration
        result[key+"_peak"] = max(row[key] for row in samples)
    return result


class MemoryWindow:
    def __init__(self, read, interval=0.05):
        if interval <= 0:
            raise ValueError("Sampling interval must be positive")
        self.read, self.interval = read, interval
        self.samples, self.errors = [], []
        self.done = threading.Event()

    def _sample(self, seconds=None):
        values = self.read()
        self.samples.append({"seconds": time.perf_counter()-self.started if seconds is None else seconds, **values})

    def _loop(self):
        try:
            while not self.done.wait(self.interval):
                self._sample()
        except Exception as error:
            self.errors.append(error)

    def start(self):
        self.started = time.perf_counter()
        self._sample(0.0)
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        return self

    def stop(self):
        elapsed = time.perf_counter()-self.started
        self.done.set()
        self.thread.join()
        if self.errors:
            raise RuntimeError("GPU sampling failed") from self.errors[0]
        # Stop at the synchronized generation boundary, not at sampler join time.
        self.samples = [row for row in self.samples if row["seconds"] < elapsed]
        self._sample(elapsed)
        return {"summary": summarize(self.samples), "samples": self.samples}


class NvidiaMemory:
    def __init__(self, index=0):
        try:
            import pynvml
        except ImportError as error:
            raise RuntimeError("Install profiling dependency: python -m pip install nvidia-ml-py") from error
        import torch
        self.nvml, self.torch = pynvml, torch
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        name = pynvml.nvmlDeviceGetName(self.handle)
        self.name = name.decode() if isinstance(name, bytes) else name
        if self.name != torch.cuda.get_device_name(0):
            raise RuntimeError("NVML GPU differs from cuda:0; set --nvml-index correctly")
        uuid = pynvml.nvmlDeviceGetUUID(self.handle)
        self.uuid = uuid.decode() if isinstance(uuid, bytes) else uuid
        torch_uuid = getattr(torch.cuda.get_device_properties(0), "uuid", None)
        if torch_uuid is not None and str(torch_uuid).removeprefix("GPU-").lower() != self.uuid.removeprefix("GPU-").lower():
            raise RuntimeError("NVML and CUDA device UUIDs disagree")

    def read(self):
        return {
            "device_used_bytes": int(self.nvml.nvmlDeviceGetMemoryInfo(self.handle).used),
            "torch_allocated_bytes": self.torch.cuda.memory_allocated(0),
            "torch_reserved_bytes": self.torch.cuda.memory_reserved(0),
        }

    def close(self):
        self.nvml.nvmlShutdown()


def aggregate(windows):
    if not windows:
        raise ValueError("No measured inference windows")
    duration = sum(w["summary"]["duration_seconds"] for w in windows)
    result = {"generation_seconds": duration, "windows": len(windows)}
    for key in ("device_used_bytes", "torch_allocated_bytes", "torch_reserved_bytes"):
        result[key+"_mean"] = sum(w["summary"][key+"_mean"]*w["summary"]["duration_seconds"] for w in windows)/duration
        result[key+"_peak"] = max(w["summary"][key+"_peak"] for w in windows)
    result["torch_peak_allocated_bytes"] = max(w["torch_peak_allocated_bytes"] for w in windows)
    result["scope"] = "Time-weighted over generation windows, including prefill and KV cache; excludes loading, warmup, tokenization and between-question I/O"
    return result
