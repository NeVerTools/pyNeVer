#include <vector>

class PropertyBounds {
private:
    std::vector<std::vector<double>> _inputMatrix;
    std::vector<double> _inputVector;
    std::vector<std::vector<double>> _outputMatrix;
    std::vector<double> _outputVector;
public:
    PropertyBounds() = default;
    PropertyBounds(std::vector<std::vector<double>> inputMatrix,
                   std::vector<double> inputVector,
                   std::vector<std::vector<double>> outputMatrix,
                   std::vector<double> outputVector);
    std::vector<std::vector<double>> getInputMatrix();
    std::vector<double> getInputVector();
    std::vector<std::vector<double>> getOutputMatrix();
    std::vector<double> getOutputVector();
};
