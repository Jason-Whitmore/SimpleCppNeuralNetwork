# SimpleCppNeuralNetwork
Simple feedforward neural network for C++.


## Motivation
There are many Neural Network libraries that exist for C++. However, many of these libraries, despite being open source, are quite difficult for beginners to understand and modify.

I've created this library to have a nice balance between simplicity and functionality. Make no mistake, this library doesn't contain features outside of normal feedforward networks, such as RNN or LSTM functionality. However, if someone wanted to add those features, they will find it easy to modify.

Structurally, this library constructs neural networks almost identically to what diagrams portray with nodes and edges, rather than with matricies. This is meant to be beginner friendly, while also simplifying many things such as gradient updates.


## Main Features
As mentioned before, this library keeps some higher level features that I've found very useful for training and testing.

### Creating Networks

Neural Networks can be created with arbitrary structure:

`std::vector<int> config = ...//index 0 for input layer size, etc.`

`NeuralNetwork n = NeuralNetwork(config);`

or conversely (Creates a network with 1 input, one output, and 2 hidden layers with 64 nodes each):

`NeuralNetwork n = NeuralNetwork({1,64,64,1});`

### Editing Networks

With a constructed neural network, you can easily edit different hyperparameters. For example, changing an activation function from LeakyRELU (default) to sigmoid can be done like this:

`NeuralNetwork n = NeuralNetwork({1,8,8,1});`

`n.nodes[3][0]->activationFunction = ActivationFunction::Sigmoid;`

This changes the output neuron to only output values in the range of (0,1), which is useful for classification tasks.

### Training Data

Training data is done through 2D vectors of doubles. Once you have populated your training inputs and outputs, they can be loaded into the network as follows:

`n.trainingInputs = inputs;`

`n.trainingOutputs = outputs;`


### Training the network

Training is done through a standard Stochastic Gradient Descent algorithm. The function has this signature:

`stochasticGradientDescent(int epochs, double learningRate)`

Hyperparameters like minibatch size and lambda for regularization are found in the function definition.

### Analyzing the network

There are a few tools to check the state of the network in order to debug.

`weightDistributionStats(double* mean, double* std)` populates the parameters with the mean and standard deviation of the weights and biases. Many claim that a normal distribution with a mean of 0 and a small standard deviation is healthy.

`maxWeight()` returns the value of the largest weight value. Many claim that if a weight is too big or small, it could mean an issue with your network.

`minWeight()` is self explanatory.

### Running the network

Once you are satisfied with your trained network. You can simply run it with the `compute(input)` function, which takes and input of vector doubles and returns a vector of doubles.

### Saving and loading networks

`loadNetwork(filename)`

`saveNetwork(filename)`

Both of these functions read/save to a simple text file with contains the parameter index then the value for each line. This means that you could manually edit weights through a text file if you desire.

### Editing network through code

I've made the code in defiance of OOP rules, meaning that everything in the network is public for client programs to use. This is really useful if you want to make modifications. For example, if you wanted to edit weights directly, you only need to access the correct vector:

`connections[index]->weight = 0`

That's it. If you have a different objective function in mind than MSE, you can easily rewrite the training and loss functions, or just edit the weights directly.


### Compiling and running

This library compiles using the standard g++ compiler avaliable on many Unix distributions. It's relevant to note that this was written using the C++11 standard. To use the library, make use to put a `#include "NeuralNetwork.h"` line at the top of your C++ file. Then just compile using g++ and make sure to use the c++11 option. Example:

`g++ NeuralNetwork.cpp example.cpp -std=c++11`


## Conclusion

This neural network library has served me as an extremely important step and tool for me to explore deep learning. I hope that the simplicity will help beginners understand how these strange models work like they helped me in the past. Please feel free to make any changes to suit your project - I certainly have made a quite a few changes for my other projects, myself.
