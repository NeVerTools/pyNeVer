#include <ilcp/cp.h>

#include "FullyConnectedNetwork.h"
#include "PropertyBounds.h"

class CPVerificationSolver {
private:
    FullyConnectedNetwork _network;
    PropertyBounds _bounds;
public:
    CPVerificationSolver(FullyConnectedNetwork network, PropertyBounds bounds);
    std::vector<IloNumExpr> matrixProduct(IloEnv env, const std::vector<std::vector<double>>& matrix, std::vector<IloNumExpr> variables);
    bool solve();
};
