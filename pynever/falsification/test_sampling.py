import glob
import sys

import numpy as np
import sampling

import logging

logger = logging.getLogger()
logger.addHandler(logging.StreamHandler(sys.stdout))
logger.setLevel(logging.INFO)

if __name__ == "__main__":
    np.random.seed(0)

    networks = glob.glob("networks/lunarlander.onnx")
    properties = glob.glob("properties/lunarlander/*.vnnlib")

    safe = []
    unsafe = []
    for network in networks:
        for prop in properties:
            logger.log(logging.INFO, prop)
            result = sampling.sampling(sampling.load_network(network), prop, n_points=5000)
            if result is None:
                unsafe.append(prop.split('/')[2])
            else:
                safe.append(prop.split('/')[2])

    print(safe)
    print(unsafe)
