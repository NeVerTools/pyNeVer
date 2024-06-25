#include <vector>

class FullyConnectedLayer {
private:
    std::vector<std::vector<double>> _weightMatrix;
    std::vector<double> _biasVector;
    bool _followedByReLU;
    int _inputSize;
    int _outputSize;
public:
    FullyConnectedLayer(std::vector<std::vector<double>> weightMatrix,
                        std::vector<double> biasVector,
                        bool followedByReLU);
    std::vector<std::vector<double>> getWeightMatrix();
    std::vector<double> getBiasVector();
    bool isFollowedByReLU();
    int getInputSize();
    int getOutputSize();
};