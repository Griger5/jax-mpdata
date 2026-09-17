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