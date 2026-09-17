import sys
import json
import subprocess
import numpy as np
import xarray as xr
from pathlib import Path
from tqdm import tqdm
import os

BASE_DIR = Path(__file__).parent.resolve()
REPO_ROOT = BASE_DIR.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

def build_docker_images():
    print("--- Building Docker Images ---")
    for directory in MODELS_DIR.iterdir():
        if not directory.is_dir() or directory.name.startswith("_"):
            continue
        
        dockerfile = directory / "Dockerfile"
        if not dockerfile.exists():
            continue
            
        image_name = f"benchmark-{directory.name.lower()}"
        print(f"Building {image_name}...")
        subprocess.run(
            ["docker", "build", "-t", image_name, "-f", str(dockerfile), str(directory)], 
            check=True, 
            stdout=subprocess.DEVNULL
        )
    print("----------------------------\n")

def run_benchmark(model_dir: Path, data_dir: Path, cores: int, iters: int, use_gpu: bool):
    image_name = f"benchmark-{model_dir.name.lower()}"
    cmd = ["docker", "run", "--rm"]

    if use_gpu:
        cmd.extend(["--gpus", "all"]) 
    else:
        cpuset = ",".join(str(i) for i in range(cores))
        cmd.extend(["--cpuset-cpus", cpuset])

    env_vars = {
        "OMP_NUM_THREADS": str(cores),
        "OPENBLAS_NUM_THREADS": str(cores),
        "MKL_NUM_THREADS": str(cores),
        "VECLIB_MAXIMUM_THREADS": str(cores),
        "NUMEXPR_NUM_THREADS": str(cores),
        "NUMBA_NUM_THREADS": str(cores),
        "NUMBA_THREADING_LAYER": "workqueue",
        "NPROC": str(cores),
        "XLA_FLAGS": f"--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads={cores} inter_op_parallelism_threads={cores}",
        "PYTHONPATH": "/repo"
    }
    
    env_args = []
    for k, v in env_vars.items():
        env_args.extend(["-e", f"{k}={v}"])
        
    # Pass the data DIRECTORY now, not a single file
    container_data_dir = "/repo/benchmarks/data"
    container_model_dir = f"/repo/benchmarks/models/{model_dir.name}"
    container_worker = "/repo/benchmarks/worker.py"

    cmd.extend([
        *env_args,
        "-v", f"{REPO_ROOT}:/repo:ro",
        image_name,
        "python", container_worker, container_data_dir, container_model_dir, str(iters)
    ])
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        raise RuntimeError(f"{model_dir.name} ({cores} threads) failed:\n{result.stderr}")
        
    return json.loads(result.stdout) # Returns dict of all datasets

if __name__ == "__main__":
    build_docker_images()
    
    # Pre-load metadata for all datasets
    dataset_paths = sorted(list(DATA_DIR.glob("*.nc")))
    timing_data = {}
    all_results = {}
    
    for data_path in dataset_paths:
        ds = xr.open_dataset(data_path)
        metadata = {k: int(ds.attrs[k]) for k in ("size_x", "size_y", "halo", "steps", "n_iters")}
        timing_data[data_path.name] = {"metadata": metadata, "data": {}}
        all_results[data_path.name] = {}

    # Loop Models -> Threads (Spawns 1 container per thread config)
    for directory in tqdm(list(MODELS_DIR.iterdir()), desc="Models"):
        if not directory.is_dir() or directory.name.startswith("_") or not (directory / "Dockerfile").exists():
            continue
        
        if os.environ.get("CI", "false").lower() == "true" and directory.name.endswith("_gpu"):
            continue
            
        config_path = directory / "config.json"
        thread_counts = [1]
        use_gpu = False
        
        if config_path.exists():
            with open(config_path) as f:
                config_data = json.load(f)
                thread_counts = config_data.get("thread_counts", [1])
                use_gpu = config_data.get("gpu", False)
            
        for cores in thread_counts:
            model_id = f"{directory.name}_{cores}t"
            if use_gpu:
                model_id += "_gpu"
                
            try:
                # Runs ALL datasets in one container!
                dataset_results = run_benchmark(directory, DATA_DIR, cores, iters=3, use_gpu=use_gpu)
            except Exception as e:
                tqdm.write(f"[ERROR] {model_id} failed: {e}")
                continue
                
            # Unpack results into the master dictionaries
            for data_name, res in dataset_results.items():
                time_results = res["times"]
                result_arr = np.array(res["result"])
                
                timing_data[data_name]["data"][model_id] = tuple(float(f"{time:.3g}") for time in time_results)
                all_results[data_name][model_id] = result_arr
                
                tqdm.write(f"  [{data_name}] {model_id} -> Min: {min(time_results):.4f}s | Avg: {sum(time_results)/len(time_results):.4f}s")

    # Final Validation & Save
    print("\n--- Validating Results ---")
    reference_algorithm = "Arabas_et_al_2014"
    total_failures = 0
    
    for data_name, results in all_results.items():
        reference_key = next((k for k in results if k.startswith(reference_algorithm)), None)
        if reference_key:
            reference = results[reference_key]
            failures = [name for name, res in results.items() if not np.allclose(res, reference, atol=5e-2, rtol=1e-5)]
            if failures:
                tqdm.write(f"[FAIL] {data_name}: {', '.join(failures)}")
                total_failures += len(failures)
                
    with open("benchmarks_results.json", "w", encoding="UTF-8") as f:
        json.dump(timing_data, f, sort_keys=True, indent=4)
        
    if total_failures:
        raise AssertionError(f"{total_failures} total algorithm validations failed.")
    print("Benchmark complete!")