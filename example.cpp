#include "NeuralNetwork.h"

int main(){
    //Neural network example here
    std::vector<int> config = std::vector<int>({1, 64, 64, 1});

    NeuralNetwork n = NeuralNetwork(config);

    std::vector<std::vector<double>> trainIn = std::vector<std::vector<double>>();
    std::vector<std::vector<double>> trainOut = std::vector<std::vector<double>>();

    for(double x = 0; x < 10; x+= 0.01){
        trainIn.push_back(std::vector<double>(1,x / 10));
        trainOut.push_back(std::vector<double>(1, (x * x)/ 100.0));
    }
    n.trainingInputs = trainIn;
    n.trainingOutputs = trainOut;
    n.randomizeNetworkUniform();


    //n.stochasticGradientDescent(0, 1000,1e-4);
    n.stochasticGradientDescent(1000, 1e-4);
    std::cout << "Loss = " << n.calculateAverageLoss() << std::endl;
    std::cout << "Max weight = " << n.getMaxParamValue() << std::endl;
    std::cout << "Min weight = " << n.getMinParamValue() << std::endl;

    double mean = 0;
    double dev = 0;

    n.getParamDistStats(&mean, &dev);

    std::cout << "Mean = " << mean << " std = " << dev << std::endl;

    std::cout << "(3.3," << n.compute(std::vector<double>(1, 3.3 / 10)).at(0) * 100 << ")" << std::endl;
    
}