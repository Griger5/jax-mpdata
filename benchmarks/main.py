from pathlib import Path
import importlib.util
import time

import xarray as xr

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

def load_module_from_path(path: Path):
    module_name = path.stem

    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module

def benchmark_module(module_name, data, metadata, iters = 100):
    results = []

    setup_function_name = "setup"
    compute_function_name = "compute"

    path = Path(module_name) / "benchmark.py"
    
    module = load_module_from_path(path)

    if hasattr(module, setup_function_name):
        setup_f = getattr(module, setup_function_name)

        setup_f(metadata)

    if hasattr(module, compute_function_name):
        compute_f = getattr(module, compute_function_name)

        for _ in range(iters):
            start = time.perf_counter()
            compute_f(data, metadata)
            end = time.perf_counter()

            results.append((end - start))

    print(f"########## {module_name.stem} ##########")
    print(f"Min = {min(results):.6f}s")
    print(f"Max = {max(results):.6f}s")
    print(f"Average = {sum(results)/len(results):.6f}s")

if __name__ == "__main__":
    for data_path in DATA_DIR.glob("*.nc"):
        ds = xr.open_dataset(data_path)

        print("####################")
        print(f"##### {data_path.stem} #####")
        print(f"size_x = {ds.attrs["size_x"]}")
        print(f"size_y = {ds.attrs["size_y"]}")
        print(f"halo = {ds.attrs["halo"]}")
        print(f"steps = {ds.attrs["steps"]}")
        print("####################")

        psi = ds["psi"].to_numpy()
        Cx = ds["Cx"].to_numpy()
        Cy = ds["Cy"].to_numpy()

        data = (psi, Cx, Cy)
        metadata = {"size_x" : ds.attrs["size_x"], "size_y" : ds.attrs["size_y"], "halo" : ds.attrs["halo"], "steps" : ds.attrs["steps"]}

        for directory in MODELS_DIR.iterdir():
            if not directory.is_dir():
                continue
            
            benchmark_module(directory, data, metadata)
