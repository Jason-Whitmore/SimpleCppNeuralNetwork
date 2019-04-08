#include <vector>
#include <stdio.h>
#include <iostream>
#include <random>
#include <fstream>
#include <math.h>
#include <thread>
#include <mutex>


enum ActivationFunction{Tanh, Sigmoid, RELU, LeakyRELU, Linear};

struct Connection;

struct Node{
    double value;
    std::vector<Connection*> inputs = std::vector<Connection*>();
    std::vector<Connection*> outputs = std::vector<Connection*>();
    ActivationFunction function;
    int id;
    double inputSum;
};

struct Connection{
    Node* start;
    Node* end;
    double weight;
    int id;
    double loss;
    bool isBias;
};


class NeuralNetwork {
    public:
    NeuralNetwork();

    NeuralNetwork(std::vector<int> layerConfig);

    ~NeuralNetwork();


    //Structure to hold the nodes in the network
    std::vector<std::vector<Node*>> nodes;

    //Training inputs. Should all be the same size and have a corresponding output
    std::vector<std::vector<double>> trainingInputs;

    //Training outputs. Should all be the same size and have a corresponding input
    std::vector<std::vector<double>> trainingOutputs;

    //Connections between the nodes. This replaces the matrix multiplication/addition found in most implementations
    std::vector<Connection*> connections;

    int numNodes;

    //Implementation detail. Use an independent node to represent bias
    Node* biasNode;

    /**Returns a random double from a uniform distribution
     * 
     * min: lower bound
     * max: upper bound
     * 
     */
    double randomDouble(double min, double max);

    /**Returns a random double using a normal distribution
     * 
     * mean: the mean of the distribution
     * stddev: the standard deviation of the distribution
     * 
     */
    static double randomDoubleNormal(double mean, double stddev);
    
    
    /**
     * Runs the current network with specified input. Similar to predict() in tensorflow.
     * 
     * Input is a vector of doubles. Although not recommended, this vector can be smaller
     * than the size of the first layer. If it's larger than the first layer, it will crash.
     * 
     * Returns a vector of doubles as output
     */
    std::vector<double> compute(std::vector<double> input);

    /**
     * Performs the dot product and activation function output.
     */
    static double getNodeOutput(Node*);

    /**
     * Returns the Loss wrt all training examples, averaged.
     */
    double calculateAverageLoss();

    /**
     * Returns the loss wrt a single training example, given by index.
     */
    double calculateLoss(int index);

    /**
     * Returns the gradient of the loss function wrt a single training example, given by index.
     */
    std::vector<double> getGradient(int index);
    
    /**
     * Returns the gradient of the loss function wrt all training examples, averaged. "True gradient"
     */
    std::vector<double> getGradient();

    /**
     * Returns the derivative of the activation function of the node evaluated at the dot product.
     */
    double getDerivative(Node* n);

    /**
     * Returns the sum of all of the outputs' loss. Used in backpropogation.
     */
    double sumNodeOutputLoss(Node*);

    /**
     * Explicitly gives the derivative of function f evaluated at input x.
     */
    double getDerivative(double x, ActivationFunction f);

    /**
     * Standard SGD training algorithm.
     * 
     * epochs is the number of times the training will do a full pass of the training examples
     * learning rate is the stepsize in which the parameters are adjusted
     * 
     * More hyperparameters inside the function definition
     */
    void stochasticGradientDescent(uint epochs, double learningRate);


    /** Calculates gradient w.r.t. multiple training examples.
     * 
     * indicies: The indicies corresponding to specific training examples
     * 
     * Returns the gradient averaged across training examples
     */
    std::vector<double> getMiniBatchGradient(std::vector<int> indicies);


    /**Splits up training indicies in preparation for minibatch SGD
     * 
     * totalNumSamples: number of training examples overall
     * minibatchSize: number of indicies per batch
     * 
     * Returns a 2d array of minibatch indicies.
     */
    static std::vector<std::vector<int>> getMinibatchIndicies(uint totalNumSamples, uint minibatchSize);


    /**
     * Returns a vector of integers that are randomized between 0 and n, without replacement
     * Used in SGD.
     */
    std::vector<int> randomOrder(int n);

    /**
     * Sets the activation function of n to function f.
     */
    void setActivationFunction(int layer, ActivationFunction f);
    
    
    /**
     * Returns the current weights/parameters as a vector of doubles.
     */
    std::vector<double> getWeights();

    /**
     * Saves the network weights to a text file with name. Format is "[weight id] [value]".
     * Example:
     * 
     * 0 3.14159
     * 1 2.719
     * 2 1.337
     */
    void saveNetwork(std::string name);
    
    /**
     * Loads the network parameters to a text file with name. Format is "[param id] [value]".
     * Example:
     * 
     * 0 3.14159
     * 1 2.719
     * 2 1.337
     */
    void loadNetwork(std::string name);

    /**
     * Returns true if s contains targetString
     */
    bool contains(std::string s, std::string targetString);


    /**
     * Splits string s into a vector of strings with delimiter splitter
     */
    std::vector<std::string> split(std::string s, std::string splitter);

    /**
     * Randomizes network parameters between (min, max) using a uniform distribution.
     */
    void randomizeNetwork(double min, double max);

    /**
     * Randomizes network parameters using a normal distribution between (-1/sqrt(i), 1/sqrt(i)) where i is the number of inputs for a node.
     */
    void randomizeNetworkUniform();

    /**
     * Returns the smallest param value for the network.
     */
    double getMinParamValue();

    /**
     * Returns the largest param value for the network.
     */
    double getMaxParamValue();

    /**
     * Populates parameters with the mean and standard distribution for the network's parameter values.
     */
    void getParamDistStats(double* mean, double* standardDeviation);


};