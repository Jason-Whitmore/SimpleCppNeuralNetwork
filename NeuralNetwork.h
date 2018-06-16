#include <vector>
#include <iostream>
#include "NodeLayer.h"
#include "Data.h"
#pragma once
class NeuralNetwork {
	public:
	NeuralNetwork(int inputs, int outputs, std::vector<int> layers);

	~NeuralNetwork();

	std::vector<double> runNetwork(std::vector<double> inputs);

	void hyperparameterOptimization(int maxNodes, double randMin, double randMax);

	void trainNetwork(double targetLoss, int maxIterations, int numOfSteps, double stepSize, double randMin, double randMax, bool displayStats);


	private:
	std::vector<NodeLayer>* layers;

	Data trainingInputs;

	Data trainingOutputs;

	double calculateCurrentLoss();

	int numWeights;

	int numBiases;

};

