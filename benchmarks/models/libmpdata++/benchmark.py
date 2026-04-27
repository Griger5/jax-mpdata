import libmpdataxx

def setup(data, metadata: dict):
    pass

def compute(data, metadata: dict):
    return libmpdataxx.mpdata_2d(data[0], data[1], data[2], 0.1, metadata["steps"], 1)

def result_to_numpy(result, metadata: dict):
    return result
