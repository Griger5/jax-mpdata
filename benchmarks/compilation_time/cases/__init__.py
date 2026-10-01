from benchmarks.compilation_time.cases.loops import jax_loop_fori, jax_loop_unrolled, numba_loop, numba_loop_parallel
from benchmarks.compilation_time.cases.stencil import jax_stencil, numba_stencil_parallel, numba_stencil_serial
from benchmarks.compilation_time.cases.control_flow import jax_control_array, jax_control_scalar, numba_control_array, numba_control_scalar, numba_control_array_parallel
from benchmarks.compilation_time.cases.recompilation import recompile_jax_stencil, recompile_numba_stencil, recompile_numba_stencil_parallel
from benchmarks.compilation_time.cases.solver import solver_first_step

registry = {
    "jax_loop_unrolled": jax_loop_unrolled,
    "jax_loop_fori": jax_loop_fori,
    "numba_loop": numba_loop,
    "numba_loop_parallel": numba_loop_parallel,

    "jax_stencil": jax_stencil,
    "numba_stencil_serial": numba_stencil_serial,
    "numba_stencil_parallel": numba_stencil_parallel,

    "jax_control_scalar": jax_control_scalar,
    "numba_control_scalar": numba_control_scalar,
    "jax_control_array": jax_control_array,
    "numba_control_array": numba_control_array,
    "numba_control_array_parallel": numba_control_array_parallel,

    "recompile_jax_stencil": recompile_jax_stencil,
    "recompile_numba_stencil": recompile_numba_stencil,
    "recompile_numba_stencil_parallel": recompile_numba_stencil_parallel,

    "solver_first_step": solver_first_step,
}