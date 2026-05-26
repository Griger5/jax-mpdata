from pathlib import Path
import importlib.util
import time
import copy
import json
import os
import platform
import sys
import subprocess

import numba

if platform.system() == "Darwin" and os.environ.get("CI", "false").lower() == "true":
    numba.set_num_threads(min(4, numba.config.NUMBA_NUM_THREADS))
else:
    numba.set_num_threads(4)

import xarray as xr
import numpy as np
from tqdm import tqdm

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

# def load_module_from_path(path: Path):
#     module_name = path.stem

#     spec = importlib.util.spec_from_file_location(module_name, path)
#     module = importlib.util.module_from_spec(spec)
#     spec.loader.exec_module(module)

#     return module

# def benchmark_module(module_name: Path, data, metadata, iters = 10):
#     time_results = []
#     result = None

#     setup_function_name = "setup"
#     compute_function_name = "compute"
#     to_numpy_function_name = "result_to_numpy"

#     path = module_name / "benchmark.py"
    
#     module = load_module_from_path(path)

#     setup_f = getattr(module, setup_function_name)
#     compute_f = getattr(module, compute_function_name)
#     to_numpy_f = getattr(module, to_numpy_function_name)

#     data_copy = copy.deepcopy(data)

#     setup_f(data_copy, metadata)

#     # avoid a cold start for JIT compilation, save a single result
#     result = compute_f(data_copy, metadata)

#     result = to_numpy_f(result, metadata)

#     for _ in range(iters):
#         data_copy = copy.deepcopy(data)
#         setup_f(data_copy, metadata)

#         start = time.perf_counter()
#         result = compute_f(data_copy, metadata)
#         result = to_numpy_f(result, metadata)
#         end = time.perf_counter()

#         time_results.append((end - start))

#     return result, time_results

WORKER_SCRIPT = """\
import sys, json, time, copy
import numpy as np
import xarray as xr
from pathlib import Path
import importlib.util
 
def load_module(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
 
data_path = Path(sys.argv[1])
module_dir = Path(sys.argv[2])
iters = int(sys.argv[3])
 
ds = xr.open_dataset(data_path)
data = (ds["psi"].to_numpy(), ds["Cx"].to_numpy(), ds["Cy"].to_numpy())
metadata = {k: int(ds.attrs[k]) for k in ("size_x", "size_y", "halo", "steps", "n_iters")}
 
mod = load_module(module_dir / "benchmark.py")

data_copy = copy.deepcopy(data)
mod.setup(data_copy, metadata)
mod.result_to_numpy(mod.compute(data_copy, metadata), metadata)
 
times = []
result = None
for _ in range(iters):
    data_copy = copy.deepcopy(data)
    mod.setup(data_copy, metadata)
    start = time.perf_counter()
    result = mod.compute(data_copy, metadata)
    result = mod.result_to_numpy(result, metadata)
    times.append(time.perf_counter() - start)
 
print(json.dumps({"times": times, "result": result.tolist()}))
"""
 
def load_module_from_path(path: Path):
    module_name = path.stem
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
 
def benchmark_module(module_dir: Path, data, metadata, data_path: Path, iters=10):
    config_path = module_dir / "config.json"
 
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        return _benchmark_subprocess(module_dir, data_path, config, iters)
 
    return _benchmark_inprocess(module_dir, data, metadata, iters)

def _benchmark_inprocess(module_dir, data, metadata, iters):
    module = load_module_from_path(module_dir / "benchmark.py")
 
    data_copy = copy.deepcopy(data)
    module.setup(data_copy, metadata)
 
    # avoid a cold start for JIT compilation, save a single result
    result = module.compute(data_copy, metadata)
    result = module.result_to_numpy(result, metadata)
 
    time_results = []
    for _ in range(iters):
        data_copy = copy.deepcopy(data)
        module.setup(data_copy, metadata)
 
        start = time.perf_counter()
        result = module.compute(data_copy, metadata)
        result = module.result_to_numpy(result, metadata)
        end = time.perf_counter()
 
        time_results.append(end - start)
 
    return result, time_results

def _benchmark_subprocess(module_dir, data_path, config, iters):
    cmd = [sys.executable, "-c", WORKER_SCRIPT, str(data_path), str(module_dir), str(iters)]
 
    if "cores" in config:
        cores = ",".join(str(i) for i in range(config["cores"]))
        cmd = ["taskset", "-c", cores] + cmd
 
    result = subprocess.run(cmd, capture_output=True, text=True)
 
    if result.returncode != 0:
        raise RuntimeError(f"{module_dir.stem} failed:\n{result.stderr}")
 
    parsed = json.loads(result.stdout)
    return np.array(parsed["result"]), parsed["times"]

if __name__ == "__main__":
    timing_data = {}

    for data_path in tqdm(list(DATA_DIR.glob("*.nc")), position = 0):
        ds = xr.open_dataset(data_path)

        tqdm.write("####################")
        tqdm.write(f"##### {data_path.stem} #####")
        tqdm.write(f"size_x = {ds.attrs["size_x"]}")
        tqdm.write(f"size_y = {ds.attrs["size_y"]}")
        tqdm.write(f"halo = {ds.attrs["halo"]}")
        tqdm.write(f"steps = {ds.attrs["steps"]}")
        tqdm.write(f"n_iters = {ds.attrs["n_iters"]}")
        tqdm.write("####################")

        psi = ds["psi"].to_numpy()
        Cx = ds["Cx"].to_numpy()
        Cy = ds["Cy"].to_numpy()

        data = (psi, Cx, Cy)
        metadata = {"size_x" : int(ds.attrs["size_x"]), "size_y" : int(ds.attrs["size_y"]), "halo" : int(ds.attrs["halo"]), "steps" : int(ds.attrs["steps"]), "n_iters" : int(ds.attrs["n_iters"])}

        results = {}

        timing_data[data_path.name] = { "metadata" : metadata, "data": {} }

        for directory in tqdm(list(MODELS_DIR.iterdir()), position = 1, leave=False):
            if not directory.is_dir() or str(directory).startswith("_"):
                continue

            if os.environ.get("CI", "false").lower() == "true":
                if str(directory).endswith("_gpu"):
                    continue
            
            result, time_results = benchmark_module(directory, data, metadata, data_path)

            results[directory.stem] = result

            timing_data[data_path.name]["data"][directory.stem] = tuple(float(f"{time:.3g}") for time in time_results)

            tqdm.write(f"########## {directory.stem} ##########")
            tqdm.write(f"Min = {min(time_results):.6f}s")
            tqdm.write(f"Max = {max(time_results):.6f}s")
            tqdm.write(f"Average = {sum(time_results)/len(time_results):.6f}s")

        reference_algorithm = "Arabas_et_al_2014"

        reference = results[reference_algorithm]
        failures = 0

        for name, res in results.items():
            if not np.allclose(res, reference, atol=5e-2, rtol=1e-5):
                tqdm.write(f"Result mismatch in \"{name}\".")
                failures += 1

        with open("benchmarks_results.json", "w", encoding="UTF-8") as f:
            json.dump(timing_data, f, sort_keys=True, indent=4)

        if failures:
            raise AssertionError(f"{failures} algorithm{"" if failures == 1 else "s"} did not match the reference result ({reference_algorithm})")
