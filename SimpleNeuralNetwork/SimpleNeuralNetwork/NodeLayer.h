#pragma once
#include <vector>
#include "Node.h"
#include "Helper.h"

class NodeLayer {
	public:
		NodeLayer();
		~NodeLayer();

		std::vector<Node> getNodes();
		void setNodes(std::vector<Node> n);
		void addNode(Node n);

		void forwardCompute();
		void backwardCompute();


	private:
		std::vector<Node> nodes;
};

