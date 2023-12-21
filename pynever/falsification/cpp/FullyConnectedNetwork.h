#include <vector>
#include "FullyConnectedLayer.h"

class FullyConnectedNetwork {
private:
    std::vector<FullyConnectedLayer> _layers;
public:
    FullyConnectedNetwork() = default;
    explicit FullyConnectedNetwork(std::vector<FullyConnectedLayer> layers);
    std::vector<FullyConnectedLayer> getLayers();
};
