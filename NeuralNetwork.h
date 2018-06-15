#include <vector>
#include <iostream>
#include "NodeLayer.h"
#pragma once
class NeuralNetwork {
	public:
	NeuralNetwork(int inputs, int outputs, std::vector<int> layers);

	~NeuralNetwork();

	std::vector<double> runNetwork(std::vector<double> inputs);

	void hyperparameterOptimization(int maxNodes, double parameterMin, double parameterMax);

	void trainNetwork();


	private:
	std::vector<NodeLayer>* layers;
};

