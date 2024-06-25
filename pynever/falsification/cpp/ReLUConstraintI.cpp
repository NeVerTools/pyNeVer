#include "ReLUConstraintI.h"

ReLUConstraintI::ReLUConstraintI(IloIntVar z, IloIntVar y)
        : IloPropagatorI(z.getEnv()), _z(z), _y(y) {
    addVar(z);
    addVar(y);
}

void ReLUConstraintI::execute () {
    if (getMax(_z) <= 0) {
        setValue(_y, 0);
    } else if (getMin(_z) > 0 && (getMin(_z) > getMin(_y))) {
        setMin(_y, getMin(_z));
        setMax(_y, getMax(_z));
    } else {
//        setMin(_y, 0);
        setMax(_y, getMax(_z));
    }

    if (getMax(_y) <= getMax(_z)) {
        setMax(_z, getMax(_y));
    }
    if ((getMin(_y) > 0) && (getMin(_y) > getMin(_z))) {
        setMin(_z, getMin(_y));
    }
}

IloPropagatorI* ReLUConstraintI::makeClone(IloEnv env) const {
    return new (env) ReLUConstraintI(_z, _y);
}

IloConstraint ReLUConstraintI::ReLU(IloIntVar z, IloIntVar y){
    return IloCustomConstraint(z.getEnv(),
                               new (z.getEnv()) ReLUConstraintI(z, y));
}