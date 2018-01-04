#pragma once
#include "Node.h"
#include "NodeLayer.h"
class NeuralNetwork {
	public:
		NeuralNetwork();
		~NeuralNetwork();

		Node pickRandomNode();


	private:
		NodeLayer* layers;

};

