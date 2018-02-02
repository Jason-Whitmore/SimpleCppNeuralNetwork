#include "NodeLayer.h"



NodeLayer::NodeLayer() {
	
	nodes = std::vector<Node>();

}



NodeLayer::~NodeLayer() {
	
}



std::vector<Node>& NodeLayer::getNodes() {
	return nodes;
}



void NodeLayer::setNodes(std::vector<Node> n) {
	nodes = n;
}



void NodeLayer::addNode(Node& n) {
	nodes.push_back(n);
}

void NodeLayer::forwardCompute() {
	double sum = 0;
	double nodeValue = 0;

	//outer loop, through nodes

	for(int n = 0; n < nodes.size(); n++) {
		
		//inner loop, through the node inputs
		for(int i = 0; i < nodes[n].getInputs().size(); i++) {
			
			sum += nodes[n].getInputs()[i].getValue() * nodes[n].getInputs()[i].getWeight();
			
		}

		//pass sum through the activation function
		nodeValue = Helper::activationFunctionRELU(sum, nodes[n].getBias());

		nodes[n].setValue(nodeValue);
		
		//apply values to the outputs
		for(int i = 0; i < nodes[n].getOutputs().size(); i++) {
			nodes[n].getOutputs()[i].setValue(nodeValue);
		}


		//debug thing

		for (int i = 0; i < nodes.size(); i++) {
			std::cout << "Node: " << i << " Value: " << nodeValue << std::endl;
		}
		
		//reset variables
		sum = 0;
		nodeValue = 0;
		
	}

}

void NodeLayer::backwardCompute() {
	double sum = 0;
	double nodeValue = 0;

	//outer loop, through nodes

	for (int n = 0; n < nodes.size(); n++) {

		//inner loop, through the node outputs
		for (int i = 0; i < nodes[n].getOutputs().size(); i++) {

			sum += nodes[n].getOutputs()[i].getValue() * nodes[n].getOutputs()[i].getWeight();

		}

		//pass sum through the activation function
		nodeValue = Helper::activationFunctionRELU(sum, nodes[n].getBias());

		nodes[n].setValue(nodeValue);

		//apply values to the inputs
		for (int i = 0; i < nodes[n].getInputs().size(); i++) {
			nodes[n].getInputs()[i].setValue(nodeValue);
		}

		//reset variables
		sum = 0;
		nodeValue = 0;

	}
}

