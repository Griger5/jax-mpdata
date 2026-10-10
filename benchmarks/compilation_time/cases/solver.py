import copy
import importlib.util
import sys
import time
from pathlib import Path

import xarray as xr

from benchmarks.compilation_time.measure import median, set_numba_threads

def solver_first_step(spec):
    target_model_dir = Path(spec["target_model_dir"])
    data_path = Path(spec["data_path"])
    warm_n = int(spec.get("warm", 3))

    # Required to load the external benchmark module dynamically
    sys.path.insert(0, str(target_model_dir))

    benchmark_path = target_model_dir / "benchmark.py"
    module_spec = importlib.util.spec_from_file_location(
        target_model_dir.stem, benchmark_path
    )
    mod = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(mod)

    threads = spec.get("threads")
    if threads is not None:
        set_numba_threads(threads)

    ds = xr.open_dataset(data_path)
    data = (ds["psi"].to_numpy(), ds["Cx"].to_numpy(), ds["Cy"].to_numpy())

    metadata = {
        k: int(ds.attrs[k])
        for k in ("size_x", "size_y", "halo", "steps", "n_iters")
    }
    metadata["steps"] = int(spec.get("steps", 1))

    def sync_result(obj):
        if hasattr(obj, "block_until_ready"):
            obj.block_until_ready()
        elif isinstance(obj, (list, tuple)):
            for o in obj: sync_result(o)
        elif isinstance(obj, dict):
            for o in obj.values(): sync_result(o)

    def run_once():
        data_copy = copy.deepcopy(data)

        start_setup = time.perf_counter()
        mod.setup(data_copy, metadata)
        setup_time = time.perf_counter() - start_setup

        start_compute = time.perf_counter()
        result = mod.compute(data_copy, metadata)
        result = mod.result_to_numpy(result, metadata)
        sync_result(result)
        compute_time = time.perf_counter() - start_compute

        return {
            "setup": setup_time,
            "compute": compute_time,
            "total": setup_time + compute_time,
        }

    first = run_once()
    warm = [run_once() for _ in range(warm_n)]
    warm_step = median([w["total"] for w in warm])

    return {
        "first_setup": first["setup"],
        "first_compute": first["compute"],
        "first_step": first["total"],
        "warm_step": warm_step,
        "compile_overhead": first["total"] - warm_step,
        "details": {"warm_runs": warm},
    }