#include "CPVerificationSolver.h"
#include <ilcp/cp.h>
#include "ReLUConstraintI.h"
#include <string>
#include <algorithm>
#include <iterator>

#define BOUND_LIMIT IlcIntMax
#define SCALE 10000

CPVerificationSolver::CPVerificationSolver(FullyConnectedNetwork network, PropertyBounds bounds) {
    this->_network = network;
    this->_bounds = bounds;
}

std::vector<IloNumExpr> CPVerificationSolver::matrixProduct(IloEnv env,
                                                            const std::vector<std::vector<double>>& matrix,
                                                            std::vector<IloNumExpr> variables) {
    std::vector<IloNumExpr> expr;
    for (const auto & row : matrix) {
        IloNumExpr e(env);
        for (int i = 0; i < row.size(); i++) {
            e += row[i] * variables[i];
        }
        expr.push_back(e);
    }
    return expr;
}

bool CPVerificationSolver::solve() {
    IloEnv env;
    bool existsSolution = false;
    try {
        IloModel model(env);

        // Input variables
        std::vector<IloIntVar> inputVars;
        for (int i = 0; i < _network.getLayers()[0].getInputSize(); i++) {
            std::string name = "I";
            name += std::to_string(i);
            inputVars.emplace_back(env, -BOUND_LIMIT, BOUND_LIMIT, name.c_str());
        }

        std::vector<IloNumExpr> inputVarsF;
        for (int i = 0; i < _network.getLayers()[0].getInputSize(); i++) {
            inputVarsF.emplace_back(inputVars[i] / SCALE);
        }

        // Bounds on the input
        std::vector<IloNumExpr> prod = matrixProduct(env, _bounds.getInputMatrix(), inputVarsF);
        for (int i = 0; i < prod.size(); i++) {
            model.add(prod[i] <= _bounds.getInputVector()[i]);
        }

        std::vector<IloIntVar> allVars;

        // Generate constraints
        std::vector<IloIntVar> currentInput = inputVars;
        std::vector<IloNumExpr> currentInputF = inputVarsF;

        for (auto var : currentInput) {
            allVars.push_back(var);
        }

        std::vector<IloIntVar> output;
        std::vector<IloNumExpr> outputF;

        int l = 1;
        for (auto layer : this->_network.getLayers()) {

            for (int i = 0; i < layer.getOutputSize(); i++) {
                std::string name = "Z";
                name += std::to_string(l);
                name += std::to_string(i);
                output.emplace_back(env, -BOUND_LIMIT, BOUND_LIMIT, name.c_str());
            }
            for (int i = 0; i < layer.getOutputSize(); i++) {
                outputF.emplace_back(output[i] / SCALE);
            }
            prod = matrixProduct(env, layer.getWeightMatrix(), currentInputF);
            for (int i = 0; i < layer.getOutputSize(); i++) {
                model.add(outputF[i] == prod[i] + layer.getBiasVector()[i]);
            }
            if (layer.isFollowedByReLU()) {
                std::vector<IloIntVar> linOutput;
                std::copy(output.begin(), output.end(), std::back_inserter(linOutput));
                for (auto var : linOutput) {
                    allVars.push_back(var);
                }

                std::vector<IloNumExpr> linOutputF;
                for (int i = 0; i < linOutput.size(); i++) {
                    linOutputF.emplace_back(linOutput[i] / SCALE);
                }

                output.clear();
                for (int i = 0; i < layer.getOutputSize(); i++) {
                    std::string name = "Y";
                    name += std::to_string(l);
                    name += std::to_string(i);
                    output.emplace_back(env, 0, BOUND_LIMIT, name.c_str());
                }
                outputF.clear();
                for (int i = 0; i < layer.getOutputSize(); i++) {
                    outputF.emplace_back(linOutput[i] / SCALE);
                }

                for (int i = 0; i < output.size(); i++) {
                    model.add(ReLUConstraintI::ReLU(linOutput[i], output[i]));
                }
                linOutput.clear();
            }
            for (auto var : output) {
                allVars.push_back(var);
            }
            currentInput.clear();
            std::copy(output.begin(), output.end(), std::back_inserter(currentInput));
            currentInputF.clear();
            for (int i = 0; i < currentInput.size(); i++) {
                currentInputF.emplace_back(currentInput[i] / SCALE);
            }
            output.clear();
            outputF.clear();

            l++;
        }
        prod = matrixProduct(env, _bounds.getOutputMatrix(), currentInputF);
        for (int i = 0; i < prod.size(); i++) {
            model.add(prod[i] <= _bounds.getOutputVector()[i]);
        }
        IloCP cp(model);
        cp.propagate();
        cp.out() << "Propagation Finished" << std::endl;
//        for (const auto& var : allVars) {
//            cp.out() << cp.domain(var) << std::endl;
//        }
//        cp.exportModel(std::cout);
        existsSolution = cp.solve();
//        if (existsSolution) {
//            for (const auto& var : allVars) {
//                cp.out() << var.getName() << " = " << cp.getValue(var) / SCALE << std::endl;
//            }
//        }
    }
    catch (IloException& ex) {
        env.out() << "Error: " << ex << std::endl;
    }
    env.end();
    return existsSolution;
}
