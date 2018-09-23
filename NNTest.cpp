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
    int id;
    double inputSum;
};

struct Connection{
    Node* start;
    Node* end;
    double weight;
    int id;
    double loss;
};



class NeuralNetwork{
    std::vector<std::vector<Node*>> nodes;

    public:std::vector<std::vector<double>> trainingInputs;

    public:std::vector<std::vector<double>> trainingOutputs;

    public:int numWeights;
    public:int numNodes;
    int currentNode;

    public:Node* biasNode;

    public:std::vector<Connection*> connections;

    public:NeuralNetwork(int numInputs, int layer1, int layer2, int numOutputs){
        numWeights = 0;
        numNodes = 0;
        //create layers
        connections = std::vector<Connection*>();
        std::vector<Node*> currentLayer = std::vector<Node*>();
        Node* n = new Node;
        n->function = ActivationFunction::Tanh;
        n->value = 1;
        n->id = numNodes;
        currentNode++;
        //create first layer
        for(int i = 0; i < numInputs; i++){
            n = new Node;
            n->function = ActivationFunction::Tanh;
            n->value = 1;
            n->id = numNodes;
            currentLayer.push_back(n);
            numNodes++;
        }
        //add layer to network
        nodes.push_back(currentLayer);
        currentLayer.clear();

        //create second layer
        for(int i = 0; i < layer1; i++){
            n = new Node;
            n->value = 0;
            n->function = ActivationFunction::LeakyRELU;
            n->id = numNodes;
            currentLayer.push_back(n);
            numNodes++;
        }

        nodes.push_back(currentLayer);
        currentLayer.clear();

        //third
        for(int i = 0; i < layer2; i++){
            n = new Node;
            n->value = 0;
            n->function = ActivationFunction::LeakyRELU;
            n->id = numNodes;
            currentLayer.push_back(n);
            numNodes++;
        }

        nodes.push_back(currentLayer);
        currentLayer.clear();

        //output
        for(int i = 0; i < numOutputs; i++){
            n = new Node;

            n->value = 0;
            n->function = ActivationFunction::LeakyRELU;
            n->id = numNodes;
            currentLayer.push_back(n);
            numNodes++;
        }
        nodes.push_back(currentLayer);
        currentLayer.clear();

        //now connect the layers
        Connection* c;
        int currentID = 0;
        //loop through layers
        for(int i = 1; i < nodes.size(); i++){
            //destination nodes
            for(int d = 0; d < nodes[i].size(); d++){
                //start nodes
                for(int s = 0; s < nodes[i - 1].size(); s++){
                    numWeights++;
                    c = new Connection();
                    c->weight = 1;
                    c->start = nodes[i-1][s];
                    c->end = nodes[i][d];
                    c->id = currentID;
                    currentID++;
                    c->start->outputs.push_back(c);
                    c->end->inputs.push_back(c);
                    connections.push_back(c);
                }
            }
        }
        //now create the bias node which connects to all nodes except for the first layer
        biasNode = new Node();
        biasNode->value = 1;

        for(int layer = 1; layer < nodes.size(); layer++){
            for(int n = 0; n < nodes[layer].size(); n++){
                numWeights++;
                c = new Connection();
                c->weight = 1;
                c->start = biasNode;
                c->end = nodes[layer][n];
                c->id = currentID;
                currentID++;
                c->start->outputs.push_back(c);
                c->end->inputs.push_back(c);
                connections.push_back(c);
            }
        }


        for(int i = 0; i < connections.size(); i++){
            connections[i]->weight = randomDouble(0,1);
        }
    }

    std::vector<double> compute(std::vector<double> inputs){
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
            outputs.push_back(nodes[nodes.size() - 1][i]->value);
        }
        return outputs;
    }

    static double getNodeOutput(Node* n){
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


    double calculateAverageLoss(){
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
                result += std::pow(std::abs(actualOutput[i] - output[i]),2) * (1.0/actualOutput.size());
            }
        }
        return result / trainingInputs.size();
    }

    double calculateLoss(int sampleIndex){
        double result = 0;
        std::vector<double> input = trainingInputs[sampleIndex];
        std::vector<double> output = trainingOutputs[sampleIndex];
        std::vector<double> actualOutput = compute(input);

        for(int i = 0; i < output.size(); i++){
            result += std::pow(actualOutput[i] - output[i], 2);
        }
        return result / output.size();
    }

    std::vector<double> getGradient(int sampleIndex){
        std::vector<double> grad = std::vector<double>(numWeights);
        //first, run the data sample through the network
        std::vector<double> output = compute(trainingInputs[sampleIndex]);

        double error;
        Connection* con;
        //set the initial error on the final weights
        for(int n = 0; n < nodes[nodes.size() - 1].size(); n++){
            error = 0.5 * std::pow(trainingOutputs[sampleIndex][n] - output[n], 2);
            //loop through the node input connections
            for(int c = 0; c < nodes[nodes.size() - 1][n]->inputs.size(); c++){
                con = nodes[nodes.size() - 1][n]->inputs[c];
                con->loss = con->start->value * getDerivative(nodes[nodes.size() - 1][n]);
                grad[con->id] = con->loss;
            }
        }


        //now that the weights have been computed, work backwards to compute the gradient wrt the loss
        for(int layer = nodes.size() - 1; layer > 0; layer--){
            for(int n = 0; n < nodes[layer].size(); n++){
                for(int in = 0; in < nodes[layer][n]->inputs.size(); in++){
                    con = nodes[layer][n]->inputs[in];
                    con->loss = con->start->value * getDerivative(nodes[nodes.size() - 1][n]) * sumNodeOutputLoss(nodes[nodes.size() - 1][n]);
                    grad[con->id] = con->loss;
                }
            }
        }

        //finally, return the gradient
        return grad;
    }

    double sumNodeOutputLoss(Node* node){
        double r = 0;
        for(int i = 0; i < node->outputs.size(); i++){
            r += node->outputs[i]->loss;
        }
        return r;
    }

    double getDerivative(Node* node){
        return getDerivative(node->value, node->function);
    }

    double getDerivative(double x, ActivationFunction f){
        if(f == ActivationFunction::LeakyRELU){
            if(x >= 0){
                return x;
            }
            return 0.1 * x;
        } else if (f == ActivationFunction::RELU){
            if(x >= 0){
                return x;
            }
            return 0;
        } else if(f == ActivationFunction::Sigmoid){
            return (1.0 / (1 + std::exp(-x))) * (1 - (1.0 / (1 + std::exp(-x))));
        } else if(f == ActivationFunction::Tanh){
            return 1 - std::pow(std::tanh(x),2);
        }


        //just to make the compiler happy...
        return 0;
    }

    void stochasticGradientDescent(double targetLoss, uint epochs, double learningRate){
        std::vector<double> gradient;
        std::vector<int> ordering;
        for(int iter = 0; iter < epochs; iter++){
            if(iter % trainingInputs.size() == 0){
                ordering = randomOrder(trainingInputs.size());
            }

            gradient = getGradient(ordering[iter]);


            if(iter % 1000 == 0 && calculateAverageLoss() < targetLoss){
                return;
            }
        }
    }

    void stochasticGradientDescentApprox(double targetLoss, int epochs, double learningRate){
        const double delta = 1e-10;
        int currentEpochs = 0;
        std::vector<double> outputs;
        std::vector<double> actualOutputs;
        std::vector<double> inputs;
        std::vector<int> order;
        int currentSample = 0;
        std::vector<double> error = std::vector<double>();
        std::vector<double> gradient = std::vector<double>(numWeights);

        int currentWeight = numWeights;
        Connection* c;
        double oldLoss;
        double newLoss;

        while(currentEpochs < epochs){

            if(currentEpochs % trainingInputs.size() == 0){
                //shuffle the data samples
                 order = randomOrder(trainingInputs.size());
                 currentSample = 0;
            }

            //feed forward the first input
            inputs = trainingInputs[order[currentSample]];
            outputs = trainingInputs[order[currentSample]];

            //loop through weights, using derivative approximations
            for(int w = 0; w < numWeights; w++){
                oldLoss = calculateLoss(order[currentSample]);
                c = connections[w];
                c->weight += delta;
                newLoss = calculateLoss(order[currentSample]);
                c->weight -= delta;
                gradient[w] = (newLoss - oldLoss) / delta;
            }

            currentSample++;
            //gradient is now calculated. Apply with learning rate

            for(int w = 0; w < numWeights; w++){
                c = connections[w];
                c->weight -= learningRate * gradient[w];
            }

            if(currentEpochs % 10 == 0 && calculateAverageLoss() < targetLoss){
                break;
            }
            if(currentEpochs % 100 == 0){
                std::cout << "New loss = " << calculateAverageLoss() << std::endl;
            }
            
            currentEpochs++;
        }

        

    }

    Node* getNode(int ID){
        for(int layer = 0; layer < nodes.size(); layer++){
            for(int node = 0; node < nodes[layer].size(); node++){
                if(nodes[layer][node]->id == ID){
                    return nodes[layer][node];
                }
            }
        }
        return nullptr;
    }

    void setActivationFunction(int ID, ActivationFunction f){
        Node* n = getNode(ID);
        n->function = f;
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

    NeuralNetwork n = NeuralNetwork(1,8,8,1);
    std::vector<double> input = std::vector<double>();

    std::vector<std::vector<double>> trainingInputs = std::vector<std::vector<double>>();
    std::vector<std::vector<double>> trainingOutputs = std::vector<std::vector<double>>();

    for(double i = -10; i < 10; i+= .01){
        trainingInputs.push_back(std::vector<double>(1,i));
        trainingOutputs.push_back(std::vector<double>(1,i * i));
    }
    n.trainingInputs = trainingInputs;
    n.trainingOutputs = trainingOutputs;
    //n.setActivationFunction()

    //std::cout << "test = " << n.compute(std::vector<double>(1,1)).at(0) << std::endl;

    n.stochasticGradientDescentApprox(0, 1e5, 1e-6);
    input.push_back(3.4);
    std::cout << "Output = " << n.compute(input).at(0) << std::endl;
    std::cout << "Weights = " << n.numWeights << std::endl;
    std::cout << "Nodes = " << n.numNodes << std::endl;
}