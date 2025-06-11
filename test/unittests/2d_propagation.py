import logging
import sys

import torch

from pynever import nodes
from pynever.networks import SequentialNetwork
from pynever.strategies.verification.algorithms import SSLPVerification, SSBPVerification
from pynever.strategies.verification.parameters import SSLPVerificationParameters, SSBPVerificationParameters
from pynever.strategies.verification.properties import VnnLibProperty

if __name__ == '__main__':
    # NETWORK DEFINITION
    W = torch.Tensor([[1, 1], [-1, 1]])

    fc_1 = nodes.FullyConnectedNode('FC_1', (2,), 2, W, torch.zeros(2))
    rl_1 = nodes.ReLUNode('ReLU_1', (2,))

    nn = SequentialNetwork('NN', 'X')
    nn.append_node(fc_1)
    nn.append_node(rl_1)

    prop = VnnLibProperty('2d_prop.vnnlib')

    logger = logging.getLogger('pynever')
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler(sys.stdout))

    print('*** SSLP ***')
    print(SSLPVerification(SSLPVerificationParameters()).verify(nn, prop))
    print()

    print('*** SSBP ***')
    print(SSBPVerification(SSBPVerificationParameters()).verify(nn, prop))
