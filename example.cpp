//Include the header file like any other library
#include "NeuralNetwork.h"

int main(){
    //Neural network example

    //Network architecture defined here
    std::vector<int> config = std::vector<int>({1, 64, 64, 1});

    //Creating the neural network
    NeuralNetwork n = NeuralNetwork(config);


    //Creating some training data
    std::vector<std::vector<double>> trainIn = std::vector<std::vector<double>>();
    std::vector<std::vector<double>> trainOut = std::vector<std::vector<double>>();

    //fitting f(x) = x^2 where x is in range [0,10). Important to have lots of data
    for(double x = 0; x < 10; x+= 0.0001){

        //Scale these inputs and outputs such that they are in the range [0,1] to prevent unstable learning.
        trainIn.push_back(std::vector<double>(1, x / 10));
        trainOut.push_back(std::vector<double>(1, (x * x)/ 100.0));
    }

    //Setting the training data
    n.trainingInputs = trainIn;
    n.trainingOutputs = trainOut;

    //Initialize network parameters randomly to help with learning
    n.randomizeNetworkUniform();

    std::cout << "Training started... (Warning: This may take a while)"
    //Train the model for 10 epochs, and print out relevant statistics after each epoch.
    for(int i = 0; i < 20; i++){
        n.stochasticGradientDescent(1, 1e-4);
        std::cout << "Iteration " << i << ":" << std::endl;
        std::cout << "Loss = " << n.calculateAverageLoss() << std::endl;
        std::cout << "Max weight = " << n.getMaxParamValue() << std::endl;
        std::cout << "Min weight = " << n.getMinParamValue() << std::endl;

        double mean = 0;
        double dev = 0;

        n.getParamDistStats(&mean, &dev);

        std::cout << "Mean = " << mean << " std = " << dev << std::endl;

        double testInput = 3.3

        std::cout << "Predicting f(" << testInput << ") = " << n.compute(std::vector<double>(1, testInput / 10)).at(0) * 100 << std::endl;
        std::cout << "Real answer:" testInput * testInput << std::endl << std::endl;
    }

    
}