import importlib
import json
from pathlib import Path

import numpy as np
import xarray as xr
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent

def resolve_path(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return path
    return BASE_DIR / path

def load_config() -> dict:
    config_path = BASE_DIR / "config.json"

    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)

def slice_halos(arr: np.ndarray, halo: int) -> np.ndarray:
    if halo == 0:
        return np.ascontiguousarray(arr)

    return np.ascontiguousarray(arr[halo:-halo, halo:-halo])

def load_case(config: dict, field_size: int, case_id: int) -> dict:
    data_dir = resolve_path(config["data_dir"])
    data_path = data_dir / f"gaussian2d_{field_size}_{case_id}.nc"

    ds = xr.open_dataset(data_path)

    psi = ds["psi"].to_numpy().astype(np.float32)
    Cx = ds["Cx"].to_numpy().astype(np.float32)
    Cy = ds["Cy"].to_numpy().astype(np.float32)

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
        "C_int": (
            slice_halos(Cx, halo),
            slice_halos(Cy, halo),
        ),
        "metadata": metadata,
    }

def make_batch(
    psi: np.ndarray,
    batch_size: int,
    different_fields: bool,
) -> np.ndarray:
    psi = np.asarray(psi, dtype=np.float32)

    if batch_size == 1 or not different_fields:
        return np.repeat(psi[None, ...], batch_size, axis=0)

    amps = np.linspace(0.9, 1.1, batch_size, dtype=np.float32)
    amps = amps[:, None, None]

    return (psi[None, ...] * amps).astype(np.float32)

def make_metrics(times: list, case: dict) -> dict:
    times = np.asarray(times, dtype=np.float64)

    batch_size = case["batch_size"]
    nx = case["metadata"]["size_x"]
    ny = case["metadata"]["size_y"]
    nt = case["metadata"]["steps"]

    t_min = float(np.min(times))
    t_max = float(np.max(times))
    t_mean = float(np.mean(times))
    t_median = float(np.median(times))

    fields = float(batch_size)
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

    benchmarks = []

    for module_name in config["implementation_modules"]:
        module = importlib.import_module(module_name)
        benchmarks.extend(module.get_benchmarks(config))

    output = {
        "config": config,
        "cases": [],
    }

    case_configs = [
        (field_size, case_id, batch_size)
        for field_size in config["field_sizes"]
        for case_id in config["case_ids"]
        for batch_size in config["batch_sizes"]
    ]

    last_case_key = None
    base_case = None

    for field_size, case_id, batch_size in tqdm(
        case_configs,
        desc="configurations",
        position=0,
        unit="cfg",
    ):
        case_key = (field_size, case_id)

        if case_key != last_case_key:
            base_case = load_case(config, field_size, case_id)
            last_case_key = case_key

        case = dict(base_case)
        case["field_size"] = field_size
        case["case_id"] = case_id
        case["batch_size"] = batch_size
        case["psi_batch"] = make_batch(
            base_case["psi"],
            batch_size,
            config["different_fields"],
        )

        case_entry = {
            "field_size": field_size,
            "case_id": case_id,
            "batch_size": batch_size,
            "data_path": base_case["data_path"],
            "results": {},
        }

        for benchmark in tqdm(
            benchmarks,
            desc=f"f={field_size}, c={case_id}, b={batch_size}",
            position=1,
            leave=False,
            unit="impl",
        ):
            name = benchmark["name"]
            run = benchmark["run"]

            times = run(case, config)
            metrics = make_metrics(times, case)
            case_entry["results"][name] = metrics

            tqdm.write(
                f"[f={field_size}, c={case_id}, b={batch_size}] "
                f"{name}: "
                f"median = {metrics['median']:.6f}s, "
                f"fields/s = {metrics['fields_per_sec']:.3f}"
            )

        output["cases"].append(case_entry)

        del case["psi_batch"]

    output_path = resolve_path(config["output"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4, sort_keys=True)

    tqdm.write("==================================================")
    tqdm.write(f"Saved results to: {output_path}")
    tqdm.write("==================================================")


if __name__ == "__main__":
    main()