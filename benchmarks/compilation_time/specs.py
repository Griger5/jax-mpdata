from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def _category(config, name):
    return config.get("categories", {}).get(name)

def _enabled(cfg, default=True):
    return bool(cfg.get("enabled", default))

def _platforms(config, cat_cfg, entry_cfg):
    platforms = entry_cfg.get(
        "platforms",
        cat_cfg.get("platforms", config.get("global", {}).get("platforms", ["cpu"])),
    )

    if isinstance(platforms, str):
        return [platforms]

    return list(platforms)

def _base_spec(config, runner, cat_cfg, entry_cfg, platform):
    global_cfg = config.get("global", {})
    spec = {"runner": runner, "platform": platform}

    if "dtype" in entry_cfg:
        spec["dtype"] = entry_cfg["dtype"]
    elif "dtype" in cat_cfg:
        spec["dtype"] = cat_cfg["dtype"]
    elif "dtype" in global_cfg:
        spec["dtype"] = global_cfg["dtype"]

    env = {
        **global_cfg.get("env", {}),
        **cat_cfg.get("env", {}),
        **entry_cfg.get("env", {}),
    }

    if env:
        spec["env"] = env

    for key in ("threads", "cores"):
        if key in entry_cfg:
            spec[key] = entry_cfg[key]
        elif key in cat_cfg:
            spec[key] = cat_cfg[key]

    return spec
def loop_specs(config):
    cat = _category(config, "loops")

    if cat is None or not _enabled(cat):
        return

    tests = cat.get("tests", {})

    default_iterations = cat.get("iterations", [])
    if isinstance(default_iterations, int):
        default_iterations = [default_iterations]

    default_shape = cat.get("shape", [1024])
    default_call_shape = cat.get("numba_call_shape", [8])
    default_warm = cat.get("warm", 3)

    for runner in ("jax_loop_unrolled", "jax_loop_fori", "numba_loop", "numba_loop_parallel"):
        test = tests.get(runner, {})

        if not _enabled(test):
            continue

        if runner.startswith("numba"):
            platforms = ["cpu"]
        else:
            platforms = _platforms(config, cat, test)

        iterations = test.get("iterations", default_iterations)
        if isinstance(iterations, int):
            iterations = [iterations]

        for platform in platforms:
            for iterations_value in iterations:
                spec = _base_spec(config, runner, cat, test, platform)
                spec["iterations"] = iterations_value

                if runner.startswith("numba"):
                    spec["call_shape"] = test.get("call_shape", default_call_shape)
                    spec["warm"] = test.get("warm", default_warm)
                else:
                    spec["shape"] = test.get("shape", default_shape)

                yield spec

def stencil_specs(config):
    cat = _category(config, "stencil")

    if cat is None or not _enabled(cat):
        return

    tests = cat.get("tests", {})

    default_shape = cat.get("shape", [256, 256])
    default_call_shape = cat.get("numba_call_shape", [4, 4])
    default_warm = cat.get("warm", 3)

    for runner in ("jax_stencil", "numba_stencil_serial", "numba_stencil_parallel"):
        test = tests.get(runner, {})

        if not _enabled(test):
            continue

        if runner == "jax_stencil":
            platforms = _platforms(config, cat, test)
        else:
            platforms = ["cpu"]

        for platform in platforms:
            spec = _base_spec(config, runner, cat, test, platform)

            if runner == "jax_stencil":
                spec["shape"] = test.get("shape", default_shape)
            else:
                spec["call_shape"] = test.get("call_shape", default_call_shape)
                spec["warm"] = test.get("warm", default_warm)

            yield spec

def control_flow_specs(config):
    cat = _category(config, "control_flow")

    if cat is None or not _enabled(cat):
        return

    tests = cat.get("tests", {})

    default_array_shape = cat.get("array_shape", [1024])
    default_call_shape = cat.get("numba_call_shape", [8])
    default_warm = cat.get("warm", 3)

    for runner in (
        "jax_control_scalar",
        "numba_control_scalar",
        "jax_control_array",
        "numba_control_array",
        "numba_control_array_parallel",
    ):
        test = tests.get(runner, {})

        if not _enabled(test):
            continue

        if runner.startswith("jax_"):
            platforms = _platforms(config, cat, test)
        else:
            platforms = ["cpu"]

        for platform in platforms:
            spec = _base_spec(config, runner, cat, test, platform)

            if runner == "jax_control_array":
                spec["shape"] = test.get("shape", default_array_shape)
            elif runner.startswith("numba_control_array"):
                spec["call_shape"] = test.get("call_shape", default_call_shape)
                spec["warm"] = test.get("warm", default_warm)
            elif runner == "numba_control_scalar":
                spec["warm"] = test.get("warm", default_warm)

            yield spec

def recompile_specs(config):
    cat = _category(config, "recompile")

    if cat is None or not _enabled(cat):
        return

    tests = cat.get("tests", {})

    default_shapes = cat.get("shapes", [])
    default_warm = cat.get("warm", 1)

    for runner in ("recompile_jax_stencil", "recompile_numba_stencil", "recompile_numba_stencil_parallel"):
        test = tests.get(runner, {})

        if not _enabled(test):
            continue

        if runner.startswith("recompile_jax"):
            platforms = _platforms(config, cat, test)
        else:
            platforms = ["cpu"]

        for platform in platforms:
            spec = _base_spec(config, runner, cat, test, platform)
            spec["shapes"] = test.get("shapes", default_shapes)
            spec["warm"] = test.get("warm", default_warm)

            yield spec

def solver_first_step_specs(config):
    cat = _category(config, "solver_first_step")

    if cat is None or not _enabled(cat):
        return

    impl_rel = config.get("paths", {}).get(
        "implementations_comparison",
        "benchmarks/implementations_comparison",
    )

    impl_dir = (PROJECT_ROOT / impl_rel).resolve()
    data_dir = impl_dir / "data"
    models_dir = impl_dir / "models"

    default_data_names = cat.get("data_names", [])
    if isinstance(default_data_names, str):
        default_data_names = [default_data_names]

    default_steps = cat.get("steps", 1)
    default_warm = cat.get("warm", 3)

    for model in cat.get("models", []):
        if not _enabled(model):
            continue

        model_name = model.get("name")
        if not model_name:
            continue

        target_model_dir = models_dir / model_name

        if not target_model_dir.exists():
            continue

        platforms = model.get("platforms", ["cpu"])
        if isinstance(platforms, str):
            platforms = [platforms]

        data_names = model.get("data_names", default_data_names)
        if isinstance(data_names, str):
            data_names = [data_names]

        for platform in platforms:
            for data_name in data_names:
                data_path = data_dir / data_name

                if not data_path.exists():
                    continue

                spec = _base_spec(config, "solver_first_step", cat, model, platform)

                spec.update(
                    {
                        "target_model_dir": str(target_model_dir),
                        "data_path": str(data_path),
                        "steps": model.get("steps", default_steps),
                        "warm": model.get("warm", default_warm),
                    }
                )

                yield spec

def all_specs(config):
    yield from loop_specs(config)
    yield from stencil_specs(config)
    yield from control_flow_specs(config)
    yield from recompile_specs(config)
    yield from solver_first_step_specs(config)