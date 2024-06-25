#include "FullyConnectedNetwork.h"

FullyConnectedNetwork::FullyConnectedNetwork(std::vector<FullyConnectedLayer> layers) {
    this->_layers = layers;
}

std::vector<FullyConnectedLayer> FullyConnectedNetwork::getLayers() {
    return this->_layers;
}
