#pragma once
#include "Node.h"
#include "NodeLayer.h"
#include "Connection.h"
#include "Helper.h"
#include <iostream>
class NeuralNetwork {
	public:
		NeuralNetwork();
		NeuralNetwork(int numInputs, int numOutputs, int layerCount, int nodesPerLayer);
		~NeuralNetwork();

		Node pickRandomNode();
		Connection pickRandomConnection();

		unsigned long long nodeCount();
		unsigned long long connectionCount();

		std::vector<std::vector<double>> getTrainingInputs();
		std::vector<std::vector<double>> getTrainingOutputs();

		void setTrainingInputs(std::vector<std::vector<double>> i);
		void setTrainingOutputs(std::vector<std::vector<double>> o);

		std::vector<double> forwardCompute(std::vector<double> inputs);
		double calculateCurrentLoss();

		void gradientDescentTraining(double targetLoss, int iterations, double lowerRandomizationBound, double upperRandomizationBound, int numberOfSteps, double stepSize);
		void gradientDescentTraining(double targetLoss, int iterations);
		void gradientDescentTraining(int iterations);




	private:
		std::vector<NodeLayer> layers;

		unsigned long long nodes;
		unsigned long long connections;

		std::vector<std::vector<double>> trainingInputs;
		std::vector<std::vector<double>> trainingOutputs;
};

