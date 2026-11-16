import json
import time
from pathlib import Path

import numpy as np
import xarray as xr
import jax
import jax.numpy as jnp

jax.config.update("jax_platform_name", "gpu")
jax.config.update("jax_enable_x64", True)

from jax_mpdata.implementation import solve

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR.parent / "results"

def resolve_path(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return path
    return BASE_DIR / path

def load_config() -> dict:
    config_path = BASE_DIR / "config.json"
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_case(config: dict, field_size: int, case_id: int) -> dict:
    data_dir = resolve_path(config["data_dir"])
    data_path = data_dir / f"gaussian2d_{field_size}_{case_id}.nc"

    ds = xr.open_dataset(data_path)

    psi = ds["psi"].to_numpy()
    Cx = ds["Cx"].to_numpy()
    Cy = ds["Cy"].to_numpy()

    halo = int(ds.attrs["halo"])

    metadata = {
        "size_x": int(ds.attrs["size_x"]),
        "size_y": int(ds.attrs["size_y"]),
        "halo": halo,
        "steps": int(config["nt"]),
        "n_iters": int(config["n_iters"]),
    }

    return {
        "data_path": str(data_path),
        "psi": psi,
        "C": (Cx, Cy),
        "metadata": metadata,
    }

def _pad_batch(psi_batch_np: np.ndarray, halo: int) -> np.ndarray:
    if halo == 0:
        return psi_batch_np
    pad_width = [(0, 0)] + [(halo, halo)] * (psi_batch_np.ndim - 1)
    return np.pad(psi_batch_np, pad_width=pad_width, mode="constant", constant_values=0.0)

def make_metrics(times: list, case: dict) -> dict:
    times = np.asarray(times, dtype=np.float64)

    nx = case["metadata"]["size_x"]
    ny = case["metadata"]["size_y"]
    nt = case["metadata"]["steps"]

    t_min = float(np.min(times))
    t_max = float(np.max(times))
    t_mean = float(np.mean(times))
    t_median = float(np.median(times))

    fields = 1.0
    cells = float(nx * ny)

    return {
        "times": [float(t) for t in times],
        "min": t_min,
        "max": t_max,
        "mean": t_mean,
        "median": t_median,
        "fields_per_sec": fields / t_median,
        "fields_per_sec_best": fields / t_min,
        "cells_per_sec": fields * cells / t_median,
        "cells_per_sec_best": fields * cells / t_min,
        "cell_steps_per_sec": fields * cells * float(nt) / t_median,
        "cell_steps_per_sec_best": fields * cells * float(nt) / t_min,
    }

def main():
    config = load_config()

    field_size = config["field_size"]
    case_id = config["case_id"]
    precisions = config["precisions"]
    warmup = int(config["warmup"])
    repeats = int(config["repeats"])

    case = load_case(config, field_size, case_id)
    metadata = case["metadata"]
    halo = metadata["halo"]
    nt = metadata["steps"]
    n_iters = metadata["n_iters"]

    psi_padded = _pad_batch(case["psi"][None, ...], halo)[0]
    Cx, Cy = case["C"]

    try:
        device = jax.devices("gpu")[0]
    except RuntimeError:
        print("ERROR: No GPU found or JAX not installed with CUDA.")
        raise

    results = {
        "config": config,
        "device": str(device),
        "device_kind": device.device_kind,
        "field_size": field_size,
        "case_id": case_id,
        "precisions": {}
    }

    for precision in precisions:
        print(f"Benchmarking {precision}...")

        try:
            jnp_dtype = getattr(jnp, precision)

            psi_jax = jax.device_put(jnp.asarray(psi_padded, dtype=jnp_dtype), device)
            Cx_jax = jax.device_put(jnp.asarray(Cx, dtype=jnp_dtype), device)
            Cy_jax = jax.device_put(jnp.asarray(Cy, dtype=jnp_dtype), device)

            # Warmup (JIT compilation + GPU cache warming)
            for _ in range(warmup):
                out = solve(psi_jax, (Cx_jax, Cy_jax), nt, halo=halo, n_iters=n_iters)
                out.block_until_ready()

            # Timed runs
            times = []
            for _ in range(repeats):
                start = time.perf_counter()
                out = solve(psi_jax, (Cx_jax, Cy_jax), nt, halo=halo, n_iters=n_iters)
                out.block_until_ready()
                times.append(time.perf_counter() - start)

            metrics = make_metrics(times, case)
            results["precisions"][precision] = metrics

            print(f"  -> Median: {metrics['median']:.6f}s | Cell-steps/sec: {metrics['cell_steps_per_sec']:.2e}")

        except Exception as e:
            print(f"  -> Failed for {precision}: {e}")
            results["precisions"][precision] = {"error": str(e)}

    output_path = resolve_path(config["output"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    print(f"Saved results to: {output_path}")

if __name__ == "__main__":
    main()