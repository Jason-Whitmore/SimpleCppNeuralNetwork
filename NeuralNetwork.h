#include <vector>
#include <iostream>
#include "NodeLayer.h"
#include "Data.h"
#include <random>
#include <cmath>
#include <fstream>
#pragma once
class NeuralNetwork {
	public:
	NeuralNetwork(int inputs, int outputs, std::vector<int> layers);

	~NeuralNetwork();

	std::vector<double> runNetwork(std::vector<double> inputs);

	std::vector<double> runNetwork(double* inputs);

	std::vector<int> hyperparameterOptimization(int maxNodes, int minNodesPerLayer, double attemptScalar, int numOfSteps, double stepSize, double randMin, double randMax);

	void trainNetwork(double targetLoss, int maxIterations, int numOfSteps, double numPassesScalar, double stepSize, double randMin, double randMax, bool displayStats);

	void optimizeRandomVariable(int numOfSteps, double stepSize, double randMin, double randMax);

	void randomizeVariables(double min, double max);

	void setTrainingInputs(Data* inputs);
	void setTrainingOutputs(Data* outputs);
	double calculateCurrentLoss();

	Data* getTrainingInputs();
	Data* getTrainingOutputs();

	void debugLayers();

	void saveWeights();
	void saveBiases();



	

	private:
	std::vector<NodeLayer>* layers;

	Data* trainingInputs;

	Data* trainingOutputs;

	std::vector<double> getAllBiases();
	std::vector<double> getAllWeights();

	void loadBiases(std::string filePath);
	void loadWeights(std::string filePath);

	double getBias(int index);
	double getWeight(int index);

	void setBias(int index, double value);
	void setWeight(int index, double value);

	int numWeights;

	int numBiases;

	int numInputs;
	int numOutputs;
};

