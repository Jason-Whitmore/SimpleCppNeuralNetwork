#include <vector>
#include <stdio.h>
#include <iostream>
#include <random>

enum ActivationFunction{Tanh, Sigmoid, RELU, LeakyRELU};

struct Connection;

struct Node{
    double value;
    std::vector<Connection*> inputs = std::vector<Connection*>();
    std::vector<Connection*> outputs = std::vector<Connection*>();
    ActivationFunction function;
};

struct Connection{
    Node* start;
    Node* end;
    double weight;
};



class NeuralNetwork{
    std::vector<std::vector<Node*>> nodes;

    std::vector<std::vector<double>> trainingInputs;

    std::vector<std::vector<double>> trainingOutputs;

    public:int numWeights;
    public:int numNodes;

    public:NeuralNetwork(int numInputs, int layer1, int layer2, int numOutputs){
        numWeights = 0;
        numNodes = 0;
        //create layers
        std::vector<Node*> currentLayer = std::vector<Node*>();
        Node* n = new Node;
        n->function = ActivationFunction::Tanh;
        n->value = 1;
        //create first layer
        for(int i = 0; i < numInputs; i++){
            n = new Node;
            n->function = ActivationFunction::Tanh;
            n->value = 1;
            currentLayer.push_back(n);
            numNodes++;
        }
        //add the bias node
        n = new Node;
        n->value = 1;
        currentLayer.push_back(n);
        numNodes++;
        //add layer to network
        nodes.push_back(currentLayer);
        currentLayer.clear();

        //create second layer
        for(int i = 0; i < layer1; i++){
            n = new Node;
            n->value = 1;
            n->function = ActivationFunction::Tanh;
            currentLayer.push_back(n);
            numNodes++;
        }
        //add the bias node
        n = new Node;
        n->value = 1;
        currentLayer.push_back(n);
        numNodes++;

        nodes.push_back(currentLayer);
        currentLayer.clear();

        //third
        for(int i = 0; i < layer2; i++){
            n = new Node;
            n->value = 1;
            n->function = ActivationFunction::Tanh;
            currentLayer.push_back(n);
            numNodes++;
        }
        //add the bias node
        n = new Node;
        n->value = 1;
        currentLayer.push_back(n);
        numNodes++;

        nodes.push_back(currentLayer);
        currentLayer.clear();

        //output
        for(int i = 0; i < numOutputs; i++){
            n = new Node;

            n->value = 1;
            n->function = ActivationFunction::LeakyRELU;
            currentLayer.push_back(n);
            numNodes++;
        }
        //add a bias node that is actually useless, but prevents a nasty bug in the next section
        n = new Node;
        n->value = 1;
        currentLayer.push_back(n);
        numNodes++;

        nodes.push_back(currentLayer);
        currentLayer.clear();

        //now connect the layers
        Connection* c;
        //loop through layers
        for(int i = 1; i < nodes.size(); i++){
            //destination nodes
            for(int d = 0; d < nodes[i].size() - 1; d++){
                //start nodes
                for(int s = 0; s < nodes[i - 1].size(); s++){
                    numWeights++;
                    c = new Connection();
                    c->weight = randomDouble(1,1);
                    c->start = nodes[i-1][s];
                    c->end = nodes[i][d];

                    c->start->outputs.push_back(c);
                    c->end->inputs.push_back(c);
                }
            }
        }
        //remove the bad bias node from earlier
        delete nodes[nodes.size() - 1][nodes[nodes.size()-1].size() - 1];
        numNodes--;
    }

    std::vector<double> compute(std::vector<double> inputs){
        std::vector<double> outputs = std::vector<double>();

        for(int i = 0; i < nodes[0].size(); i++){
            nodes[0][i]->value = inputs[i];
        }

        //loop each layer
        for(int layer = 1; layer < nodes.size(); layer++){
            //loop each node in layer
            for(int n = 0; n < nodes[layer].size() - 1; n++){
                nodes[layer][n]->value = getNodeOutput(nodes[layer][n]);
            }
        }

        for(int i = 0; i < nodes[nodes.size() - 1].size(); i++){
            outputs.push_back(nodes[nodes.size() - 1][i]->value);
        }
        return outputs;
    }

    static double getNodeOutput(Node* n){
        double dotProduct = 0;

        for(int i = 0; i < n->inputs.size(); i++){
            dotProduct += n->inputs[i]->weight * n->inputs[i]->start->value;
        }

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
            if(dotProduct > 0){
                return dotProduct;
            } else {
                return dotProduct * 0.1;
            }
        } else if (function == ActivationFunction::Sigmoid){
            return 1.0 / (1 + std::exp(-dotProduct));
        }
        //just in case
        return 0;
    }

    double randomDouble(double min, double max){
        double scalar = (double)rand() / RAND_MAX;
        return min + (scalar * (max - min));
    }

    void debugWeights(){
        int weightNum = 0;
        for(int layer = 1; layer < nodes.size(); layer++){
            for(int n = 0; n < nodes[layer].size(); n++){
                for(int c = 0; c < nodes[layer][n]->inputs.size(); c++){
                    std::cout << "Weight " << weightNum << ": " << nodes[layer][n]->inputs[c]->weight << std::endl;
                    weightNum++;
                }

                
            }
        }
    }

    double calculateAverageError(){
        double result = 0;

        std::vector<double> input;
        std::vector<double> output;
        std::vector<double> actualOutput;
        //loop through the training data
        for(int dp = 0; dp < trainingInputs.size(); dp++){
            input = trainingInputs[dp];
            output = trainingOutputs[dp];

            actualOutput = compute(input);

            //loop through each output feature
            for(int i = 0; i < actualOutput.size(); i++){
                result += std::abs(actualOutput[i] - output[i]) * (1.0/actualOutput.size());
            }
        }
        return result / trainingInputs.size();
    }

    void stochasticGradientDescent(double targetError, int epochs, double learningRate){

        int currentEpochs = 0;
        std::vector<double> outputs;
        std::vector<double> actualOutputs;
        std::vector<double> inputs;
        std::vector<int> order;
        int currentSample = 0;
        std::vector<double> error = std::vector<double>();
        std::vector<double> gradient = std::vector<double>(numWeights);

        int currentWeight = numWeights;

        while(currentEpochs < epochs){

            if(currentEpochs >= trainingInputs.size()){
                //shuffle the data samples
                 order = randomOrder(trainingInputs.size());
                 currentSample = 0;
            }

            //feed forward the first input
            inputs = trainingInputs[order[currentSample]];
            outputs = trainingInputs[order[currentSample]];
            actualOutputs = compute(inputs);

            //calculate error in the last layer
            for(int i = 0; i < outputs.size(); i++){
                error.push_back((outputs[i] - actualOutputs[i]) * actualOutputs[i] * (1 - actualOutputs[i]));
            }

            //loop through the hidden layers backwards, calculating the gradients
            for(int layer = nodes.size() - 1; layer > 0; layer--){
                for(int node = nodes[layer].size(); node >= 0; node--){
                    for(int connection = nodes[layer][node]->inputs.size(); connection >=0; connection-- ){
                        
                        currentWeight--;
                    }
                }
            }

            currentSample++;
        }

        
    }


    std::vector<double> getWeights(){
        std::vector<double> r = std::vector<double>();

        for(int layer = 1; layer < nodes.size(); layer++){
            for(int node = 0; node < nodes[layer].size() - 1; node++){
                for(int connection = 0; connection < nodes[layer][node]->inputs.size(); connection++){
                    r.push_back(nodes[layer][node]->inputs[connection]->weight);
                }
            }
        }
        return r;
    }

    std::vector<int> randomOrder(int size){
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

    

};



int main(){
    std::cout << "Hello World!" << std::endl;

    NeuralNetwork n = NeuralNetwork(1,8,8,1);
    std::vector<double> input = std::vector<double>();
    n.debugWeights();
    input.push_back(1);
    std::cout << "Output = " << n.compute(input).at(0) << std::endl;
    std::cout << "Weights = " << n.numWeights << std::endl;
    std::cout << "Nodes = " << n.numNodes << std::endl;
}