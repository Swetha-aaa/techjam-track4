"""Latency and memory profile.

The submission rules ask for latency to be disclosed. We have no model to load
and no network calls to make, so the figures are worth reporting explicitly
rather than leaving to inference.

Two memory numbers are reported. `tracemalloc` sees only Python-level
allocations, and our FTS5 index lives inside SQLite's C layer — so the Python
figure understates the real footprint badly. Peak resident set size is the honest
number and the one we report.

Run: python -m eval.perf
"""
import ctypes
import time
import tracemalloc

from evaluator.local_evaluator import evaluate, load_jsonl, catalog_index
from starter.agent import Agent

SESSIONS = load_jsonl("data/public_set.jsonl")
IDS, CATS, PRODS = catalog_index("data/catalog.jsonl")


def peak_rss_mb():
    """Peak resident set size in MB. Tries psutil, then platform APIs."""
    try:
        import psutil
        info = psutil.Process().memory_info()
        return getattr(info, "peak_wset", info.rss) / 1e6
    except ImportError:
        pass
    try:                                        # POSIX
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    except ImportError:
        pass
    try:                                        # Windows fallback
        import ctypes.wintypes

        class PMC(ctypes.Structure):
            _fields_ = [("cb", ctypes.wintypes.DWORD),
                        ("PageFaultCount", ctypes.wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t),
                        ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t),
                        ("PeakPagefileUsage", ctypes.c_size_t)]

        c = PMC()
        c.cb = ctypes.sizeof(c)
        ok = ctypes.windll.kernel32.K32GetProcessMemoryInfo(
            ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(c), c.cb)
        return c.PeakWorkingSetSize / 1e6 if ok else 0.0
    except Exception:
        return 0.0


def main():
    rss_before = peak_rss_mb()
    tracemalloc.start()

    t0 = time.perf_counter()
    agent = Agent()
    startup = time.perf_counter() - t0
    _, py_index = tracemalloc.get_traced_memory()
    rss_index = peak_rss_mb()

    latencies = []
    original = agent._recommend

    def timed(st, top_k):
        t = time.perf_counter()
        out = original(st, top_k)
        latencies.append((time.perf_counter() - t) * 1000)
        return out

    agent._recommend = timed

    t0 = time.perf_counter()
    result = evaluate(agent, SESSIONS, IDS, CATS, PRODS)
    wall = time.perf_counter() - t0
    _, py_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    rss_peak = peak_rss_mb()

    latencies.sort()
    n = len(latencies)

    print("STARTUP  (once per process)")
    print(f"  FTS5 index build over 50,000 products   {startup:8.2f} s")
    print(f"  peak RSS after index build              {rss_index:8.1f} MB")
    print(f"  ...of which Python-level allocations    {py_index / 1e6:8.1f} MB")
    print()
    print(f"PER-TURN LATENCY  (n = {n} turns)")
    print(f"  mean                                    {sum(latencies) / n:8.2f} ms")
    print(f"  median                                  {latencies[n // 2]:8.2f} ms")
    print(f"  p95                                     {latencies[int(n * 0.95)]:8.2f} ms")
    print(f"  p99                                     {latencies[int(n * 0.99)]:8.2f} ms")
    print(f"  max                                     {max(latencies):8.2f} ms")
    print()
    print("FULL RUN  (200 sessions)")
    print(f"  wall clock, excluding startup           {wall:8.2f} s")
    print(f"  peak RSS                                {rss_peak:8.1f} MB")
    print(f"  baseline RSS before agent construction  {rss_before:8.1f} MB")
    print(f"  LLM tokens consumed                     "
          f"{result['reported_token_usage']['total_tokens']:8d}")
    print()
    print("  dependencies : none beyond the Python standard library")
    print("  network      : not required")
    print("  model weights: none")


if __name__ == "__main__":
    main()