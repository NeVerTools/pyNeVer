#include "FullyConnectedLayer.h"

FullyConnectedLayer::FullyConnectedLayer(std::vector<std::vector<double>> weightMatrix,
                                         std::vector<double> biasVector,
                                         bool followedByReLU) {
    this->_weightMatrix = weightMatrix;
    this->_biasVector = biasVector;
    this->_followedByReLU = followedByReLU;
    this->_outputSize = weightMatrix.size();
    this->_inputSize = weightMatrix[0].size();
}

std::vector<std::vector<double>> FullyConnectedLayer::getWeightMatrix() {
    return this->_weightMatrix;
}

std::vector<double> FullyConnectedLayer::getBiasVector() {
    return this->_biasVector;
}

bool FullyConnectedLayer::isFollowedByReLU() {
    return this->_followedByReLU;
}

int FullyConnectedLayer::getInputSize() {
    return this->_inputSize;
}

int FullyConnectedLayer::getOutputSize() {
    return this->_outputSize;
}
