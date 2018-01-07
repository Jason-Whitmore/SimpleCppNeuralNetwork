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

