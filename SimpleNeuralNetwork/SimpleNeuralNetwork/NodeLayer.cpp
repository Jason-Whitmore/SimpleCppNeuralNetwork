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
