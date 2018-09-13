#include <vector>
#include <stdio.h>
#include <iostream>
#include <random>

enum ActivationFunction{Tanh, Sigmoid, RELU, LeakyRELU};

struct Node{
    double value;
    struct Connection;
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

    NeuralNetwork(int numInputs, int layer1, int layer2, int numOutputs){
        //create layers
        std::vector<Node*> currentLayer = std::vector<Node*>();
        Node* n;
        n->function = ActivationFunction::Tanh;
        n->value = 1;
        //create first layer
        for(int i = 0; i < numInputs; i++){
            n = new Node;
            n->value = 1;
            currentLayer.push_back(n);
        }
        //add the bias node
        n = new Node;
        n->value = 1;
        currentLayer.push_back(n);
        //add layer to network
        nodes.push_back(currentLayer);
        currentLayer.clear();

        //create second layer
        for(int i = 0; i < layer1; i++){
            n = new Node;
            n->value = 1;
            n->function = ActivationFunction::Tanh;
            currentLayer.push_back(n);
        }
        //add the bias node
        n = new Node;
        n->value = 1;
        currentLayer.push_back(n);

        nodes.push_back(currentLayer);
        currentLayer.clear();

        //third
        for(int i = 0; i < layer2; i++){
            n = new Node;
            n->value = 1;
            n->function = ActivationFunction::Tanh;
            currentLayer.push_back(n);
        }
        //add the bias node
        n = new Node;
        n->value = 1;
        currentLayer.push_back(n);

        nodes.push_back(currentLayer);
        currentLayer.clear();

        //output
        for(int i = 0; i < numOutputs; i++){
            n = new Node;
            n->value = 1;
            n->function = ActivationFunction::LeakyRELU;
        }

        //now connect the layers
        Connection* c;
        //loop through layers
        for(int i = 1; i < 4; i++){
            //destination nodes
            for(int d = 0; d < nodes[i].size() - 1; d++){
                //start nodes
                for(int s = 0; s < nodes[i - 1].size(); s++){
                    c = new Connection;
                    c->weight = randomDouble(-1,1);
                    c->start = nodes[i-1][s];
                    c->end = nodes[i][d];

                    (nodes[i-1][s])->outputs.push_back(c);
                    nodes[i][d]->inputs.push_back(c);
                }
            }
        }


    }

    double randomDouble(double min, double max){
        double scalar = (double)rand() / RAND_MAX;

        return min + (scalar * (max - min));
    }


};



int main(){
    std::cout << "Hello World!" << std::endl;
}