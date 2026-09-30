import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from tqdm import tqdm

from benchmarks.compilation_time.specs import all_specs

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parents[1]

def load_config(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def run_spec(spec, config):
    global_cfg = config.get("global", {})

    worker_module = global_cfg.get(
        "worker_module",
        "benchmarks.compilation_comparison.worker",
    )

    use_taskset = global_cfg.get("taskset", True)

    with tempfile.TemporaryDirectory(prefix="compilation_bench_") as tmp:
        tmp_path = Path(tmp)

        env = os.environ.copy()
        env["JAX_COMPILATION_CACHE_DIR"] = str(tmp_path / "jax_cache")
        env["NUMBA_CACHE_DIR"] = str(tmp_path / "numba_cache")

        threads = spec.get("threads")
        if threads is not None:
            env["NUMBA_NUM_THREADS"] = str(threads)
            env["OMP_NUM_THREADS"] = str(threads)

        for key, value in spec.get("env", {}).items():
            env[key] = str(value)

        cmd = [sys.executable, "-m", worker_module]

        cores = spec.get("cores")
        if use_taskset and cores is not None and shutil.which("taskset"):
            core_list = ",".join(str(i) for i in range(int(cores)))
            cmd = ["taskset", "-c", core_list] + cmd

        proc = subprocess.run(
            cmd,
            input=json.dumps(spec),
            capture_output=True,
            text=True,
            env=env,
            cwd=PROJECT_ROOT,
        )

        stdout = proc.stdout.strip()
        stderr = proc.stderr.strip()

        parsed_stdout = None

        if stdout:
            try:
                parsed_stdout = json.loads(stdout)
            except json.JSONDecodeError:
                parsed_stdout = None

        if proc.returncode != 0:
            if isinstance(parsed_stdout, dict) and parsed_stdout.get("traceback"):
                raise RuntimeError(parsed_stdout["traceback"])

            if stderr:
                raise RuntimeError(stderr)

            if stdout:
                raise RuntimeError(stdout)

            raise RuntimeError("worker failed without stdout or stderr")

        if parsed_stdout is None:
            raise RuntimeError(f"could not parse worker output:\n{stdout}")

        return parsed_stdout

def save_results(records, results_path):
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=4, sort_keys=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(PACKAGE_DIR / "config.json"),
    )
    args = parser.parse_args()

    config = load_config(args.config)
    global_cfg = config.get("global", {})

    results_dir = Path(global_cfg.get("results_dir", "results"))
    if not results_dir.is_absolute():
        results_dir = PACKAGE_DIR / results_dir

    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / "compilation_time.json"

    records = []
    failed_count = 0

    specs = list(all_specs(config))

    for spec in tqdm(specs):
        tqdm.write(spec["runner"])

        try:
            result = run_spec(spec, config)
        except Exception as exc:
            failed_count += 1

            result = {
                "error": str(exc),
            }

            tqdm.write(f"FAILED {spec['runner']}", file=sys.stderr)
            tqdm.write(str(exc), file=sys.stderr)

        records.append(
            {
                "spec": spec,
                "result": result,
            }
        )

        save_results(records, results_path)

    if failed_count:
        tqdm.write(f"{failed_count} benchmark(s) failed", file=sys.stderr)
        raise SystemExit(1)

if __name__ == "__main__":
    main()