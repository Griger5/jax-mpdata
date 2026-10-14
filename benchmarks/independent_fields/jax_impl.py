import time

import jax
import numpy as np

from jax_mpdata.permutations import solve, solve_batch

def _pad_batch(psi_batch_np: np.ndarray, halo: int) -> np.ndarray:
    if halo == 0:
        return psi_batch_np

    pad_width = [(0, 0)] + [(halo, halo)] * (psi_batch_np.ndim - 1)

    return np.pad(
        psi_batch_np,
        pad_width=pad_width,
        mode="constant",
        constant_values=0.0,
    )

def _run_jax(device, batched: bool, case: dict, config: dict) -> list:
    metadata = case["metadata"]

    nt = metadata["steps"]
    halo = metadata["halo"]
    n_iters = metadata["n_iters"]

    C_np = case["C"]

    # case["psi_batch"] is the interior field.
    # JAX solve/solve_batch expect halo cells to be present.
    psi_batch_np = _pad_batch(case["psi_batch"], halo)

    batch_size = psi_batch_np.shape[0]

    repeats = int(config["repeats"])
    warmup = max(1, int(config["warmup"]))

    C = (
        jax.device_put(C_np[0], device),
        jax.device_put(C_np[1], device),
    )

    if batched:
        template = jax.device_put(psi_batch_np, device)

        for _ in range(warmup):
            inp = template.copy()
            out = solve_batch(
                inp,
                C,
                nt,
                halo=halo,
                n_iters=n_iters,
            )
            out.block_until_ready()

        times = []

        for _ in range(repeats):
            inp = template.copy()

            start = time.perf_counter()

            out = solve_batch(
                inp,
                C,
                nt,
                halo=halo,
                n_iters=n_iters,
            )
            out.block_until_ready()

            times.append(time.perf_counter() - start)

        return times

    template = [
        jax.device_put(psi_batch_np[i], device)
        for i in range(batch_size)
    ]

    def run_once() -> float:
        inputs = [x.copy() for x in template]

        start = time.perf_counter()

        for i in range(batch_size):
            psi_i = inputs[i]
            inputs[i] = None

            out = solve(
                psi_i,
                C,
                nt,
                halo=halo,
                n_iters=n_iters,
            )
            out.block_until_ready()

        return time.perf_counter() - start

    for _ in range(warmup):
        run_once()

    return [run_once() for _ in range(repeats)]

def get_benchmarks(config: dict) -> list:
    benchmarks = []

    for device_name in config["jax_devices"]:
        device = jax.devices(device_name)[0]

        for mode in config["jax_modes"]:
            batched = mode == "batched"
            name = f"jax_{device_name}_{mode}"

            def run(case, config, device=device, batched=batched):
                return _run_jax(device, batched, case, config)

            benchmarks.append(
                {
                    "name": name,
                    "run": run,
                }
            )

    return benchmarks