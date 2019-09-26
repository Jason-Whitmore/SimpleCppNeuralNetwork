#include "NeuralNetwork.h"



NeuralNetwork::NeuralNetwork(){

}


NeuralNetwork::NeuralNetwork(std::vector<int> layerConfig){
    srand(time(NULL));
    numNodes = 0;

    nodes = std::vector<std::vector<Node*>>();
    //construct the layers
    for(int layer = 0; layer < layerConfig.size(); layer++){
        std::vector<Node*> currentLayer = std::vector<Node*>();

        //std::cout << layerConfig[layer] << std::endl;
        for(int n = 0; n < layerConfig[layer]; n++){
            Node* node = new Node;
            
            if(layer == 0 || layer == layerConfig.size() - 1){
                node->function = ActivationFunction::Linear;
            } else {
                node->function = ActivationFunction::LeakyRELU;
            }
            node->value = 0;
            currentLayer.push_back(node);

        }
        nodes.push_back(currentLayer);

    }

    //connect layers together with weights
    int currentID = 0;
    for(int layer = 1; layer < nodes.size(); layer++){
        for(int nodeCurrent = 0; nodeCurrent < nodes[layer].size(); nodeCurrent++){
            for(int nodePrev = 0; nodePrev < nodes[layer - 1].size(); nodePrev++){
                Connection* c = new Connection;

                c->id = currentID;
                c->isBias = false;
                connections.push_back(c);

                //make sure pointers are arranged correctly for the connection, start, end nodes
                Node* start = nodes[layer - 1][nodePrev];
                Node* end = nodes[layer][nodeCurrent];

                start->outputs.push_back(c);
                end->inputs.push_back(c);
                c->start = start;
                c->end = end;

                currentID++;
            }
        }
    }


    //now create the one bias node to be used in the network.
    Node* biasNode = new Node;
    biasNode->value = 1;
    biasNode->function = ActivationFunction::Linear;
    
    //connect to every node except for the input nodes
    for(int layer = 1; layer < nodes.size(); layer++){
        for(int n = 0; n < nodes[layer].size(); n++){
            Connection* c = new Connection;
            c->id = currentID;
            c->isBias = true;

            connections.push_back(c);

            Node* end = nodes[layer][n];
            biasNode->outputs.push_back(c);
            end->inputs.push_back(c);
            c->isBias = true;
            c->start = biasNode;
            c->end = end;
            
            currentID++;
        }
    }

}

NeuralNetwork::~NeuralNetwork(){
    //delete all nodes
    for(int l = 0; l < nodes.size(); l++){
        for(int n = 0; n < nodes[l].size(); n++){
            delete nodes[l][n];
        }
    }

    //delete connections
    for(int i = 0; i < connections.size(); i++){
        delete connections[i];
    }
}


std::vector<double> NeuralNetwork::compute(std::vector<double> inputs){
    std::vector<double> outputs = std::vector<double>();

    for(int i = 0; i < nodes[0].size(); i++){
        nodes[0][i]->value = inputs[i];
    }

    //loop each layer
    for(int layer = 1; layer < nodes.size(); layer++){
        //loop each node in layer
        for(int n = 0; n < nodes[layer].size(); n++){
            nodes[layer][n]->value = getNodeOutput(nodes[layer][n]);
        }
    }

    //retrieve output
    for(int i = 0; i < nodes[nodes.size() - 1].size(); i++){
        outputs.push_back(getNodeOutput(nodes[nodes.size() - 1][i]));
    }
    return outputs;
}

double NeuralNetwork::getNodeOutput(Node* n){
    if(n->inputs.size() == 0){
        return n->value;
    }

    double dotProduct = 0;

    for(int i = 0; i < n->inputs.size(); i++){
        dotProduct += n->inputs[i]->weight * n->inputs[i]->start->value;
    }
    
    n->inputSum = dotProduct;

    //Apply activation function
    ActivationFunction function = n->function;
    if(function == ActivationFunction::Tanh){
        return tanh(dotProduct);
    } else if(function == ActivationFunction::RELU){
        if(dotProduct > 0){
            return dotProduct;
        } else {
            return 0;
        }
    } else if (function == ActivationFunction::LeakyRELU){
        if(dotProduct >= 0){
            return dotProduct;
        } else {
            return dotProduct * 0.01;
        }
    } else if (function == ActivationFunction::Sigmoid){
        return 1.0 / (1 + std::exp(-dotProduct));
    } else if(function == ActivationFunction::Linear){
        return dotProduct;
    }
    //just in case
    return 0;
}

double NeuralNetwork::randomDouble(double min, double max){
    double scalar = ((double)rand()) / RAND_MAX;
    return min + (scalar * (max - min));
}


double NeuralNetwork::calculateAverageLoss(){
    double result = 0;

    //loop through the training data
    for(int dp = 0; dp < trainingInputs.size(); dp++){
        result += calculateLoss(dp);
    }
    return result / trainingInputs.size();
}

double NeuralNetwork::calculateLoss(int sampleIndex){
    double result = 0;
    std::vector<double> input = trainingInputs[sampleIndex];
    std::vector<double> output = trainingOutputs[sampleIndex];
    std::vector<double> actualOutput = compute(input);

    for(int i = 0; i < output.size(); i++){
        result += std::pow(std::abs(actualOutput[i] - output[i]),2);
    }
    return result / output.size();
}

std::vector<double> NeuralNetwork::getGradient(int sampleIndex){

    const double delta = 1e-6;
    const double clip = 1;
    std::vector<double> grad = std::vector<double>(connections.size());
    //first, run the data sample through the network
    std::vector<double> output = compute(trainingInputs[sampleIndex]);

    double error;
    Connection* con;
    //set the initial error on the final weights
    for(int n = 0; n < nodes[nodes.size() - 1].size(); n++){
        //loop through the node input connections
        for(int c = 0; c < nodes[nodes.size() - 1][n]->inputs.size(); c++){
            con = nodes[nodes.size() - 1][n]->inputs[c];
            int id = con->id;
            double scalar = 1.0 / nodes[nodes.size() - 1].size();

            connections[id]->loss = 2 * (output[n] - trainingOutputs[sampleIndex][n]) *
                                    con->start->value * getDerivative(con->end) * scalar;

            //clip the gradients to help stablize learning
            if(con->loss > clip){
                con->loss = clip;
            }
            if(con->loss < -clip){
                con->loss = -clip;
            }

            grad[con->id] = con->loss;
            
        }
    }


    //now that the weights have been computed, work backwards to compute the gradient wrt the loss
    for(int layer = nodes.size() - 2; layer > 0; layer--){
        for(int n = 0; n < nodes[layer].size(); n++){
            for(int in = 0; in < nodes[layer][n]->inputs.size(); in++){
                con = nodes[layer][n]->inputs[in];

                
                
                con->loss = con->start->value * getDerivative(nodes[layer][n]) * sumNodeOutputLoss(nodes[layer][n]);
                
                if(con->loss > clip){
                    con->loss = clip;
                }
                if(con->loss < -clip){
                    con->loss = -clip;
                }
                grad[con->id] = con->loss;
            }

        }
    }

    //finally, return the gradient
    return grad;
}


double NeuralNetwork::sumNodeOutputLoss(Node* node){
    double r = 0;
    for(int i = 0; i < node->outputs.size(); i++){
        r += node->outputs[i]->loss;
    }
    return r;

}

double NeuralNetwork::getDerivative(Node* node){
    return getDerivative(node->inputSum, node->function);
}

double NeuralNetwork::getDerivative(double x, ActivationFunction f){
    if(f == ActivationFunction::LeakyRELU){
        if(x >= 0){
            return 1;
        }
        return 0.01;
    } else if (f == ActivationFunction::RELU){
        if(x >= 0){
            return 1;
        }
        return 0;
    } else if(f == ActivationFunction::Sigmoid){
        return (1.0 / (1 + std::exp(-x))) * (1 - (1.0 / (1 + std::exp(-x))));
    } else if(f == ActivationFunction::Tanh){
        return 1 - std::pow(std::tanh(x),2);
    } else if(f == ActivationFunction::Linear){
        return 1;
    }


    //just to make the compiler happy...
    std::cout << "Warning. Returning bad derivative." << std::endl;
    return 0;
}

std::vector<std::vector<int>> NeuralNetwork::getMinibatchIndicies(uint totalNumSamples, uint minibatchSize){
    //generate a vector with all of the indicies
    std::vector<int> indiciesRemaining = std::vector<int>();

    for(int i = 0; i < totalNumSamples; i++){
        indiciesRemaining.push_back(i);
    }

    std::vector<std::vector<int>> minibatchIndicies = std::vector<std::vector<int>>();

    std::vector<int> currentMiniBatch = std::vector<int>();

    while(indiciesRemaining.size() > 0){
        //randomly pull from vector of indicies left
        int index = rand() % indiciesRemaining.size();
        int number = indiciesRemaining[index];

        //remove it fully from vector
        indiciesRemaining.erase(indiciesRemaining.begin() + index);

        currentMiniBatch.push_back(number);

        //done with current minibatch?
        if(currentMiniBatch.size() >= minibatchSize){
            minibatchIndicies.push_back(currentMiniBatch);

            currentMiniBatch.clear();
        }
        
    }

    if(currentMiniBatch.size() > 0){
        minibatchIndicies.push_back(currentMiniBatch);
    }

    return minibatchIndicies;
}


std::vector<double> NeuralNetwork::getMiniBatchGradient(std::vector<int> indicies){
    std::vector<double> r = std::vector<double>(connections.size());

    //For each training example in batch
    for(int i = 0; i < indicies.size(); i++){
        std::vector<double> tempGrad = getGradient(indicies[i]);

        //Average together gradients
        for(int n = 0; n < connections.size(); n++){
            r[n] += (1.0 / indicies.size()) * tempGrad[n];
        }
    }

    return r;
}




void NeuralNetwork::stochasticGradientDescent(uint epochs, double learningRate){

    //more hyperparameters to adjust
    const double lambda = 0.00;
    const uint minibatchSize = 32;

    for(uint e = 0; e < epochs; e++){
        //start of epoch, shuffle sample indicies

        std::vector<std::vector<int>> batchIndicies = getMinibatchIndicies(trainingInputs.size(), minibatchSize);

        //make updates when batches are calculated
        for(int b = 0; b < batchIndicies.size(); b++){

            std::vector<double> miniBatchGrad = getMiniBatchGradient(batchIndicies[b]);

            //apply to parameters
            for(int i = 0; i < miniBatchGrad.size(); i++){
                connections[i]->weight = connections[i]->weight - (learningRate * miniBatchGrad[i]) - (lambda * learningRate * connections[i]->weight);
            }
        }
    }
}




void NeuralNetwork::setActivationFunction(int layerNum, ActivationFunction f){

    for(int i = 0; i < nodes[layerNum].size(); i++){
        nodes[layerNum][i]->function = f;
    }
}


std::vector<double> NeuralNetwork::getWeights(){
    std::vector<double> r = std::vector<double>();

    for(int w = 0; w < connections.size(); w++){
        r.push_back(connections[w]->weight);
    }

    return r;
}

std::vector<int> NeuralNetwork::randomOrder(int size){
    std::vector<int> r = std::vector<int>();
    for(int i = 0; i < size; i++){
        r.push_back(i);
    }

    int temp;
    int index1;
    int index2;

    //Shuffling loop. Not ideal but it's fairly efficient
    for(int i = 0; i < size * 2; i++){
        //find two elements
        index1 = (int)randomDouble(0, size);
        index2 = (int)randomDouble(0, size);
        
        //swap
        temp = r[index1];
        r[index1] = r[index2];
        r[index2] = temp;
    }

    return r;
}

void NeuralNetwork::saveNetwork(std::string fileName){
    std::string contents = "";

    for(uint w = 0; w < connections.size(); w++){
        contents += std::to_string(w) + " " + std::to_string(connections[w]->weight) + "\n";
    }
    //file ready to be saved
    std::ofstream file;
    file.open(fileName);
    file << contents;
    file.close();
}

bool NeuralNetwork::contains(std::string s, std::string targetString) {
    int targetLength = targetString.length();

    int currentPosition = 0;

    while (currentPosition + targetLength <= s.length()) {
        if (s.substr(currentPosition,targetLength) == targetString) {
            return true;
        }
        currentPosition++;
    }

    return false;
}

std::vector<std::string> NeuralNetwork::split(std::string s, std::string splitter) {
    std::vector<std::string> r = std::vector<std::string>();

    //Make copy of the string so we don't destroy the original.
    std::string copy = s;
    
    //While there is still splitter characters in the string copy
    while (contains(copy, splitter)) {
        //add substring to return vector
        r.push_back(copy.substr(0, copy.find_first_of(splitter)));
        copy = copy.substr(copy.find_first_of(splitter) + splitter.length());
    }

    r.push_back(copy);


    return r;
    
}

void NeuralNetwork::loadNetwork(std::string fileName){
    std::ifstream file(fileName);

    std::string singleLine;
    std::vector<std::string> lineSeparated;

    if(file.is_open()){
        while(std::getline(file, singleLine)){
            lineSeparated = split(singleLine, " ");

            int index = std::stoi(lineSeparated[0]);
            double weight = std::stod(lineSeparated[1]);

            connections[index]->weight = weight;
        }
    }

    file.close();
}

void NeuralNetwork::randomizeNetwork(double min, double max){
    for(int i = 0; i < connections.size(); i++){
        if(connections[i]->isBias){
            connections[i]->weight = 0;
        } else {
            connections[i]->weight = randomDouble(min,max);
        }
    }
}


double NeuralNetwork::randomDoubleNormal(double mean, double stddev){
    if(stddev == 0){
        return mean;
    }
    std::default_random_engine gen;
    std::normal_distribution<double> distribution(mean, stddev);

    return distribution(gen);
}

double NeuralNetwork::getMinParamValue(){

    double min = connections[0]->weight;

    for(int i = 1; i < connections.size(); i++){
        if(connections[i]->weight < min){
            min = connections[i]->weight;
        }
    }

    return min;
}

double NeuralNetwork::getMaxParamValue(){

    double max = connections[0]->weight;

    for(int i = 1; i < connections.size(); i++){
        if(connections[i]->weight > max){
            max = connections[i]->weight;
        }
    }

    return max;    
}

void NeuralNetwork::getParamDistStats(double* mean, double* standardDeviation){
    //first, calculate the mean
    double sum = 0;
    for(int i = 0; i < connections.size(); i++){
        sum += connections[i]->weight;
    }

    *mean = sum / connections.size();

    //use calculated mean to find the standard deviation

    double variance = 0;

    for(int i = 0; i < connections.size(); i++){
        variance += std::pow(connections[i]->weight - *mean, 2);
    }
    variance /= connections.size();

    *standardDeviation = std::sqrt(variance);
}

void NeuralNetwork::randomizeNetworkUniform(){

    for(int l = 1; l < nodes.size(); l++){
        for(int n = 0; n < nodes[l].size(); n++){
            int inputNum = nodes[l][n]->inputs.size();

            for(int in = 0; in < nodes[l][n]->inputs.size(); in++){

                //Commonly used equation for parameter initialization
                nodes[l][n]->inputs[in]->weight = randomDouble(-1.0/ std::sqrt(inputNum), 1.0/ std::sqrt(inputNum));
            }
        }
    }
}