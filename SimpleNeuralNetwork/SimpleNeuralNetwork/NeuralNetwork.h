#pragma once
#include "RELUNode.h"
#include "NodeLayer.h"
#include "Connection.h"
#include "Helper.h"
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
class NeuralNetwork {
	public:
		NeuralNetwork(int numInputs, int numOutputs, int layerCount, int nodesPerLayer);
		~NeuralNetwork();

		Node& pickRandomNode();
		Connection* pickRandomConnection();

		unsigned long long getNodeCount();
		unsigned long long getConnectionCount();

		std::vector<std::vector<double>>& getTrainingInputs();

		std::vector<std::vector<double>>& getTrainingOutputs();

		void setTrainingInputs(std::vector<std::vector<double>> i);
		void setTrainingInputs(std::string fileName, std::string entrySeparator, char pointSeperator);

		void setTrainingOutputs(std::vector<std::vector<double>> o);
		void setTrainingOutputs(std::string fileName, std::string entrySeparator, char pointSeperator);

		std::vector<double> forwardCompute(std::vector<double> inputs);
		double calculateCurrentLoss();

		void gradientDescentTraining(double targetLoss, int iterations, double lowerRandomizationBound, double upperRandomizationBound, int numberOfSteps, double stepSize);
		void gradientDescentTraining(double targetLoss, int iterations);
		void gradientDescentTraining(int iterations);


		void loadWeights();
		void loadBiases();

		void saveBiases();
		void saveWeights();

		
		void readInputsFromCSV();
		void readOutputsFromCSV();

		void testMethod();

		void debugWeights();
		void debugBiases();

	private:
		std::vector<NodeLayer> layers;

		unsigned long long nodes;
		unsigned long long connections;

		std::vector<std::vector<double>> trainingInputs;
		std::vector<std::vector<double>> trainingOutputs;

		void optimizeBias(Node n, int steps, double stepSize);
		void optimizeWeight(Connection* c, int steps, double stepSize);
		void randomizeAllVariables(double min, double max);

		Node& getNode(unsigned long long index);
		Connection* getConnection(unsigned long long index);

		double extractDoubleFromString(std::string s);

		void cleanOutNetwork();

		
};

