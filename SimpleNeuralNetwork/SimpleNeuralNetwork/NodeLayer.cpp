#include "NodeLayer.h"



NodeLayer::NodeLayer() {
	

}



NodeLayer::~NodeLayer() {
	
}



std::vector<Node> NodeLayer::getNodes() {
	return nodes;
}



void NodeLayer::setNodes(std::vector<Node> n) {
	nodes = n;
}



void NodeLayer::addNode(Node n) {
	nodes.push_back(n);
}

void NodeLayer::forwardCompute() {
	double sum = 0;

	//outer loop, through nodes

	for(int n = 0; n < nodes.size(); n++) {
		
		//inner loop, through the node inputs
		for(int i = 0; i < nodes[n].getInputs().size(); i++) {
			
			
			
		}


	}

}

void NodeLayer::backwardCompute() {

}
