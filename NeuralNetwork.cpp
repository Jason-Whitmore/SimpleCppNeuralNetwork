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

    numWeights = connections.size();
    srand(time(NULL));

}
/**
NeuralNetwork NeuralNetwork::neuralNetworkInit(std::vector<int> layerConfig){
    return NeuralNetwork::NeuralNetwork(layerConfig);
}

NeuralNetwork NeuralNetwork::neuralNetworkInit(int numInputs, int numNodesLayer1, int numNodesLayer2, int numOutputs){
    std::vector<int> config = std::vector<int>();

    config.push_back(numInputs);
    config.push_back(numNodesLayer1);
    config.push_back(numNodesLayer2);
    config.push_back(numOutputs);
    
    return NeuralNetwork::NeuralNetwork(config);
}

static NeuralNetwork neuralNetworkInit(int numInputs, int numNodesLayer1, int numOutputs){
    std::vector<int> config = std::vector<int>();

    config.push_back(numInputs);
    config.push_back(numNodesLayer1);
    config.push_back(numOutputs);
    
    return NeuralNetwork::NeuralNetwork(config);
}

**/

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

    //activation function
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
    std::vector<double> grad = std::vector<double>(numWeights);
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

            /**
            double lossBefore = calculateLoss(sampleIndex);
            con->weight += delta;
            double lossAfter = calculateLoss(sampleIndex);
            con->weight -= delta;

            con->loss = (lossAfter - lossBefore) / delta;
            **/
            if(con->loss > 10){
                con->loss = 10;
            }
            if(con->loss < -10){
                con->loss = -10;
            }

            grad[con->id] = con->loss;
            
        }
    }


    //now that the weights have been computed, work backwards to compute the gradient wrt the loss
    for(int layer = nodes.size() - 2; layer > 0; layer--){
        for(int n = 0; n < nodes[layer].size(); n++){
            for(int in = 0; in < nodes[layer][n]->inputs.size(); in++){
                //connection is either a bias or a weight
                if(nodes[layer][n]->inputs[in]->isBias){
                    //bias
                } else {
                    //weight
                }
                con = nodes[layer][n]->inputs[in];

                //double lossWRToutput = con->start->value * getDerivative(nodes[layer][n]) * sumNodeOutputLoss(nodes[layer][n]);
                
                
                con->loss = con->start->value * getDerivative(nodes[layer][n]) * sumNodeOutputLoss(nodes[layer][n]);
                //std::cout << "weight " << std::to_string(con->id) << " = " << con->loss << std::endl;
                
                if(con->loss > 10){
                    con->loss = 10;
                }
                if(con->loss < -10){
                    con->loss = -10;
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

void NeuralNetwork::stochasticGradientDescent(double targetLoss, uint epochs, double learningRate){
    std::vector<double> gradient;
    std::vector<int> ordering;

    const double lambda = 0.0;
    std::string csvString = "";
    

    double effectiveLearningRate = learningRate;
    uint currentSample = 0;
    for(int iter = 0; iter < epochs * trainingInputs.size(); iter++){
        if(iter % trainingInputs.size() == 0){
            ordering = randomOrder(trainingInputs.size());
            currentSample = 0;
        }

        gradient = getGradient(ordering[currentSample]);
        //apply learning rate to gradient
        Connection* c;
        for(int i = 0; i < gradient.size(); i++){
            c = connections[i];
            c->weight = c->weight - (learningRate * gradient[i]) - learningRate * lambda * c->weight;
        }


        if(iter % trainingInputs.size() == 0 && calculateAverageLoss() < targetLoss){
            return;
        }
        if(iter % trainingInputs.size() == 0){
            std::cout << iter/trainingInputs.size() << " Loss = " << calculateAverageLoss() << std::endl;
            csvString += std::to_string(iter/ trainingInputs.size()) + "," + std::to_string(calculateAverageLoss()) + "\n";
            //std::cout << "Avg gradient slope = " << gradientAvgAbsValue(gradient) << std::endl;
        }
        currentSample++;

        std::ofstream file;
        file.open("gradData.csv");

        file << csvString;
        file.close();
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

    std::string copy = s;
    
    while (contains(copy, splitter)) {
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
            //connections[i]->weight = randomDoubleNormal(0,0.001);
        }
    }
}


double NeuralNetwork::randomDoubleNormal(double mean, double variance){
    std::default_random_engine gen;
    std::normal_distribution<double> distribution(mean, variance);

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
                nodes[l][n]->inputs[in]->weight = randomDouble(-1.0/ std::sqrt(inputNum), 1.0/ std::sqrt(inputNum));
            }
        }
    }
}

int main(){
    //Neural network example here
    std::vector<int> config = std::vector<int>({1,128,128, 1});

    NeuralNetwork n = NeuralNetwork(config);

    std::vector<std::vector<double>> trainIn = std::vector<std::vector<double>>();
    std::vector<std::vector<double>> trainOut = std::vector<std::vector<double>>();

    for(double x = 0; x < 10; x+= 0.001){
        trainIn.push_back(std::vector<double>(1,x / 10.0));
        trainOut.push_back(std::vector<double>(1, (x * x)/ 100.0));
    }
    n.trainingInputs = trainIn;
    n.trainingOutputs = trainOut;
    n.randomizeNetworkUniform();



    n.stochasticGradientDescent(0.0001, 100000, 1e-4);
    std::cout << "Max weight = " << n.getMaxParamValue() << std::endl;
    std::cout << "Min weight = " << n.getMinParamValue() << std::endl;

    double mean = 0;
    double dev = 0;

    n.getParamDistStats(&mean, &dev);

    std::cout << "Mean = " << mean << " std = " << dev << std::endl;

    std::cout << "(3.3," << n.compute(std::vector<double>(1,3.3 / 10.0)).at(0) * 100 << ")" << std::endl;
    
}