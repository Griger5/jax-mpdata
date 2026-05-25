from benchmarks.models.PyMPDATA_4threads.benchmark import _setup

solver = None

def setup(data, metadata: dict):
    global solver

    solver = _setup(data, metadata, 1)    

def compute(data, metadata: dict):
    solver.advance(n_steps=metadata["steps"])

def result_to_numpy(result, metadata: dict):
    return solver.advectee.get()