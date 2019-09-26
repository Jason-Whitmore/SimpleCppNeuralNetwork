#include <vector>
#include <stdio.h>
#include <iostream>
#include <random>
#include <fstream>
#include <math.h>
#include <thread>
#include <mutex>

/*
 * Enumuration to represent each type of activation function.
 */
enum ActivationFunction{Tanh, Sigmoid, RELU, LeakyRELU, Linear};

//Forward declaring a Connection struct so it can be used in the Node struct definition.
struct Connection;

/*
 * Struct that represents a single node in the neural network, as opposed to using strictly linear algebra.
 */
struct Node{
    double value;
    std::vector<Connection*> inputs = std::vector<Connection*>();
    std::vector<Connection*> outputs = std::vector<Connection*>();
    ActivationFunction function;
    int id;
    double inputSum;
};

/*
 *Complete Connection struct definition. Note the start and end Node pointers. The weight value will be optimized during training.
 */
struct Connection{
    Node* start;
    Node* end;
    double weight;
    int id;
    double loss;
    bool isBias;
};

/*
 * The primary class that contains the Neural network data structure, along with the various functions to train, make predictions, and debug.
 */
class NeuralNetwork {
    public:
    NeuralNetwork();

    /*
     * The primary constructor for creating a neural network.
     * 
     * layerConfig: A vector of integers specifying the structure of the network. Index 0 is the input layer, while the last element is the output layer 
     */
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

    /**Gets a random double from a uniform distribution
     * 
     * min: lower bound of the distribution.
     * max: upper bound of the distribution.
     * 
     * Returns the random double
     */
    static double randomDouble(double min, double max);

    /**Gets a random double from a normal distribution
     * 
     * mean: the mean of the distribution
     * stddev: the standard deviation of the distribution
     * 
     * Returns the random double
     */
    static double randomDoubleNormal(double mean, double stddev);
    
    
    /**
     * Runs the current network with specified input. Similar to predict() in tensorflow.
     * 
     * input: The input vector of the network. Should be the same size of the first layer.
     * 
     * Returns a vector of doubles as output
     */
    std::vector<double> compute(std::vector<double> input);

    /**
     * Performs the dot product and activation function output.
     *
     * Node*: The node to retrieve the output from.
     *
     * Returns the output for the node
     */
    static double getNodeOutput(Node* n);

    /**
     * Calculates the loss with respect to all training examples, averaged. Warning: Can be very slow on massive datasets.
     *
     * Returns the loss as a double
     */
    double calculateAverageLoss();

    /**
     * Calculates the loss with respect to a single training example.
     *
     * index: The index of the training example to calculate loss from.
     *
     * Returns the loss as a double
     */
    double calculateLoss(int index);

    /**
     * Gets the gradient of the loss function with respect to a single training example.
     *
     * index: The index of the training example to calculate the gradient from.
     *
     * Returns the gradient as a vector of doubles.
     */
    std::vector<double> getGradient(int index);
    
    /**
     * Gets the gradient of the loss function with respect to all training examples, averaged. "True gradient". Extremely slow.
     * 
     * Returns the gradient as a vector of doubles.
     */
    std::vector<double> getGradient();

    /**
     * Calculates the derivative of the activation function of the node evaluated at the dot product.
     *
     * n: The target node
     *
     * Returns the derivative as a double
     */
    double getDerivative(Node* n);

    /**
     * Sums all of the outputs' loss. Used in backpropogation.
     *
     * n: The target node
     *
     * Returns the sum as a double
     */
    double sumNodeOutputLoss(Node* n);

    /**
     * Explicitly gives the derivative of function evaluated at a certain input
     *
     * x: The input to evaluate the derivative at
     * f: The activation function.
     *
     * Returns the derivative as a double
     */
    double getDerivative(double x, ActivationFunction f);

    /**
     * Standard SGD training algorithm.
     * 
     * epochs: The number of times the training will do a full pass of the training examples
     * learningRate: the stepsize in which the parameters are adjusted in scalar-vector multiplication with a gradient.
     * 
     * Note: More hyperparameters inside the function definition
     */
    void stochasticGradientDescent(uint epochs, double learningRate);


    /** Calculates gradient with respect to multiple training examples.
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
     * Gets a vector containing a random shuffling of integers from 0 to n.
     * Used in SGD.
     *
     * n: the maximum integer to be found in the vector
     *
     * Returns the random order as a vector of integers.
     */
    std::vector<int> randomOrder(int n);

    /**
     * Sets the activation function of a layer.
     *
     * layer: The integer index of the layer
     * f: The activation function that the layer's nodes will be set to.
     *
     */
    void setActivationFunction(int layer, ActivationFunction f);
    
    
    /**
     * Gets the weights of the neural network
     *
     * Returns the weights as a vector of doubles, with the indicies corresponding with the connection ids.
     */
    std::vector<double> getWeights();

    /**
     * Saves the network weights to a text file with name. Format is "[weight id] [value]".
     * Example:
     * 
     * 0 3.14159
     * 1 2.719
     * 2 1.337
     *
     * name: The name of the file to write to.
     */
    void saveNetwork(std::string name);
    
    /**
     * Loads the network parameters to a text file with name. Format is "[param id] [value]".
     * Example:
     * 
     * 0 3.14159
     * 1 2.719
     * 2 1.337
     *
     * name: The name of the file to read from 
     */
    void loadNetwork(std::string name);

    /**
     * Helper function that determines if a string occured in another string
     *
     * 
     * s: The string where targetString will be searched in.
     * targetString: The string that will be searched for.
     *
     * Returns true if targetString is found within s, else false
     */
    bool contains(std::string s, std::string targetString);


    /**
     * Splits string using a delimiter/splitter
     *
     * s: The string to be broken apart
     * splitter: The delimiter or boundary which separates the string out
     *
     * Returns a vector of the strings that were separated.
     */
    std::vector<std::string> split(std::string s, std::string splitter);

    /**
     * Randomizes network parameters using a uniform distribution.
     *
     * min: The lower bound for the distribution
     * max: The upper bound for the distribution
     */
    void randomizeNetwork(double min, double max);

    /**
     * Randomizes network parameters using a normal distribution between (-1/sqrt(i), 1/sqrt(i)) where i is the number of inputs for a node.
     */
    void randomizeNetworkUniform();

    /**
     * Returns the smallest param value for the network.
     *
     * Returns the param as a double
     */
    double getMinParamValue();

    /**
     * Returns the largest param value for the network.
     *
     * Returns the param as a double
     */
    double getMaxParamValue();

    /**
     * Calculates the mean (mu) and standard deviation (sigma) of the weights and biases
     *
     * mean: Pointer whose value will be populated with the mean
     * standardDeviation: Pointer whose value will be populated with the standard deviation
     */
    void getParamDistStats(double* mean, double* standardDeviation);


};