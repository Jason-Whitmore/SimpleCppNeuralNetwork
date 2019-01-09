# SimpleCppNeuralNetwork
Simple feedforward neural network for C++.


## Motivation
There are many Neural Network libraries that exist for C++. However, many of these libraries, despite being open source, are quite difficult for beginners to understand and modify.

I've created this library to have a nice balance between simplicity and functionality. Make no mistake, this library doesn't contain features outside of normal feedforward networks, such as RNN or LSTM functionality. However, if someone wanted to add those features, they will find it easy to modify.

Structurally, this library constructs neural networks almost identically to what diagrams portray with nodes and edges, rather than with matricies. This is meant to be beginner friendly, while also simplifing many things such as gradient updates.

Overall, this library should be useful to beginners and to C++ programmers who find heavy deep learning libraries inappropriate for their projects.

## Main Features
As mentioned before, this library keeps some higher level features that I've found very useful for training and testing.

### Creating Networks

Neural networks can be created in several ways:

First, a simple 3 layer network (1 hidden layer):

`NeuralNetwork n = NeuralNetwork(3,16,1)`

Second, a simple 4 layer network (2 hidden layers):

`NeuralNetwork n = NeuralNetwork(3,16,16,1)`

Finally, an arbitrarily sized neural network:

`std::vector<int> config = //index 0 for input layer size, etc.`

`NeuralNetwork n = NeuralNetwork(config)`

### Editing Networks

With a constructed neural network, you can easily edit different hyperparameters. For example, changing an activation function from LeakyRELU (default) to sigmoid can be done like this:

`NeuralNetwork n = NeuralNetwork(1,8,8,1);`

`n.nodes[3][0]->activationFunction = ActivationFunction::Sigmoid;`

This changes the output neuron to only output values in the range of (0,1), which is useful for classification tasks.

### Training Data

Training data is done through 2D vectors of doubles. These can be loaded manually, or written from a CSV file. Once you have populated your training inputs and outputs, they can be loaded into the network as follows:

`n.trainingInputs = inputs`

`n.trainingOutputs = outputs`

Additionally, you can also supply testing data, which will help when determining if your network is overfitting your data.

`n.testInputs = testIn`

`n.testOutputs = testOut`

### Training the network

Training is done through a standard Stochastic Gradient Descent algorithm. There are many overloaded variants of this function to accomodate your needs. The generic function has arguments like this:

`stochasticGradientDescent(double targetLoss, int epochs, double learningRate, boolean verbose, boolean recordProgress)`

Every epoch, the network will check the loss against the targetLoss. If it's less, then it will finish training early, else it will keep training another epoch, or until the max number of epochs is hit.

The "verbose" argument is very similar to Tensorflow's verbose option: if set to true, it will print out training (and testing, if applicable) loss every epoch.

"recordProgress" will write the verbose text out to a .csv file so you can visualize the progress of training over time in your favorite data visualizer (I personally plot the points in LibreOffice Calc ;) ).

### Analyzing the network

There are a few tools to check the state of the network in order to debug.

`weightDistributionStats(double* mean, double* std)` populates the parameters with the mean and standard deviation of the weights and biases. Many claim that a distribution of N(0,1) is healthy.

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


## Conclusion

This neural network library has served me as an extremely important step and tool for me to explore deep learning. I hope that the simplicity will help beginners understand how these strange models work like they helped me in the past. Likewise, I think this library will be useful to more experienced hobbyists who are looking for something simpler and lighter than the more popular libraries. Please feel free to make any changes to suit your project - I certainly have made a quite a few changes for my other projects, myself.
