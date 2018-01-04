#pragma once
#include <vector>
#include "Node.h"


class NodeLayer {
	public:
		NodeLayer();
		~NodeLayer();

		std::vector<Node> getNodes();
		void setNodes(std::vector<Node> n);
		void addNode(Node n);

	private:
		std::vector<Node> nodes;
};

