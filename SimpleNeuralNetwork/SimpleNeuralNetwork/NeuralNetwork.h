#pragma once
#include "Node.h"
#include "NodeLayer.h"
#include "Connection.h"
#include "Helper.h"
#include <iostream>
class NeuralNetwork {
	public:
		NeuralNetwork();
		~NeuralNetwork();

		Node pickRandomNode();
		Connection pickRandomConnection();

		unsigned long long nodeCount();
		unsigned long long connectionCount();

	private:
		std::vector<NodeLayer> layers;

		unsigned long long nodes;
		unsigned long long connections;
};

