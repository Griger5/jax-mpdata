from models.Arabas_et_al_2014 import mpdata

solver = None

def setup(data, metadata: dict = {}):
    global solver
    solver = mpdata.solver_donorcell(
        bcx = mpdata.cyclic,
        bcy = mpdata.cyclic,
        nx = metadata["size_x"],
        ny = metadata["size_y"],
    )

def compute(data, metadata: dict = {}):
    solver.state()[:] = data[0]
    solver.courant(0)[:,:] = data[1][metadata["halo"]:-metadata["halo"], :]
    solver.courant(1)[:,:] = data[2][:, metadata["halo"]:-metadata["halo"]]
    solver.solve(metadata["steps"])

    return solver.state()