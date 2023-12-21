#include "PropertyBounds.h"

PropertyBounds::PropertyBounds(std::vector<std::vector<double>> inputMatrix,
                               std::vector<double> inputVector,
                               std::vector<std::vector<double>> outputMatrix,
                               std::vector<double> outputVector) {
    this->_inputMatrix = inputMatrix;
    this->_inputVector = inputVector;
    this->_outputMatrix = outputMatrix;
    this->_outputVector = outputVector;
}

std::vector<std::vector<double>> PropertyBounds::getInputMatrix() {
    return this->_inputMatrix;
}

std::vector<double> PropertyBounds::getInputVector() {
    return this->_inputVector;
}

std::vector<std::vector<double>> PropertyBounds::getOutputMatrix() {
    return this->_outputMatrix;
}

std::vector<double> PropertyBounds::getOutputVector() {
    return this->_outputVector;
}
