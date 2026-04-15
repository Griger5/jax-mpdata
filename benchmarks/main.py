from pathlib import Path
import importlib.util
import time

import xarray as xr
import numpy as np

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

def load_module_from_path(path: Path):
    module_name = path.stem

    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module

def benchmark_module(module_name: Path, data, metadata, iters = 10):
    time_results = []
    result = None

    setup_function_name = "setup"
    compute_function_name = "compute"
    to_numpy_function_name = "result_to_numpy"

    path = module_name / "benchmark.py"
    
    module = load_module_from_path(path)

    setup_f = getattr(module, setup_function_name)
    compute_f = getattr(module, compute_function_name)
    to_numpy_f = getattr(module, to_numpy_function_name)

    setup_f(data, metadata)

    # avoid a cold start for JIT compilation, save a single result
    result = compute_f(data, metadata)

    result = to_numpy_f(result, metadata)

    for _ in range(iters):
        setup_f(data, metadata)

        start = time.perf_counter()
        result = compute_f(data, metadata)
        result = to_numpy_f(result, metadata)
        end = time.perf_counter()

        time_results.append((end - start))

    return result, time_results

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

        results = {}

        for directory in MODELS_DIR.iterdir():
            if not directory.is_dir() or str(directory).startswith("_"):
                continue
            
            result, time_results = benchmark_module(directory, data, metadata)

            results[directory.stem] = result

            print(f"########## {directory.stem} ##########")
            print(f"Min = {min(time_results):.6f}s")
            print(f"Max = {max(time_results):.6f}s")
            print(f"Average = {sum(time_results)/len(time_results):.6f}s")

        reference_algorithm = "Arabas_et_al_2014"

        reference = results[reference_algorithm]
        failures = 0

        for name, res in results.items():
            if not np.allclose(res, reference, atol=1e-6, rtol=1e-5):
                print(f"Result mismatch in \"{name}\".")
                failures += 1

        if failures:
            raise AssertionError(f"{failures} algorithm{"" if failures == 1 else "s"} did not match the reference result ({reference_algorithm})")
