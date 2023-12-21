#include <ilcp/cpext.h>

class ReLUConstraintI : IloPropagatorI {
private:
    IloIntVar _z;
    IloIntVar _y;
public:
    explicit ReLUConstraintI(IloIntVar z, IloIntVar y);
    void execute() override;
    IloPropagatorI * makeClone(IloEnv env) const override;
    static IloConstraint ReLU(IloIntVar z, IloIntVar y);
};
