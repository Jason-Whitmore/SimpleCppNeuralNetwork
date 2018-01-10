#include "NeuralNetwork.h"


NeuralNetwork::NeuralNetwork(){
	
}

NeuralNetwork::NeuralNetwork(int numInputs, int numOutputs, int layerCount, int nodesPerLayer) {

}


NeuralNetwork::~NeuralNetwork(){
	
}

Node NeuralNetwork::pickRandomNode() {
	int nodeNumber = Helper::randomNumber(0, nodeCount());
	
	int currentNode = 0;
	for (int l = 0; l < layers.size(); l++) {
		
		for (int n = 0; n < layers[l].getNodes().size(); n++) {
			if (currentNode != nodeNumber) {
				currentNode++;
			} else {
				return layers[l].getNodes()[n];
			}
		}

	}
	

}

Connection NeuralNetwork::pickRandomConnection() {
	int connectionNumber = Helper::randomNumber(0, connectionCount());
	int currentConnection = 0;

	for (int l = 1; l < layers.size(); l++) {
		
		for (int n = 0; n < layers[l].getNodes().size(); n++) {
			for(int c = 0; c < layers[l].getNodes()[n].getInputs().size(); c++) {
				if (currentConnection != connectionNumber) {
					currentConnection++;
				} else {
					return layers[l].getNodes()[n].getInputs()[c];
				}
			}
		}
	}
}

unsigned long long NeuralNetwork::nodeCount() {
	return nodes;
}

unsigned long long NeuralNetwork::connectionCount() {
	return connections;
}

std::vector<std::vector<double>> NeuralNetwork::getTrainingInputs() {
	return trainingInputs;
}

std::vector<std::vector<double>> NeuralNetwork::getTrainingOutputs() {
	return trainingOutputs;
}

void NeuralNetwork::setTrainingInputs(std::vector<std::vector<double>> i) {
	trainingInputs = i;
}

void NeuralNetwork::setTrainingOutputs(std::vector<std::vector<double>> o) {
	trainingOutputs = o;
}

std::vector<double> NeuralNetwork::forwardCompute(std::vector<double> inputs) {
	if (inputs.size() != layers[0].getNodes().size()) {
		//exception here
	}
	//move inputs to the first layer of nodes

	for (int i = 0; i < layers[0].getNodes().size(); i++) {
		layers[0].getNodes()[i].setValue(inputs[i]);
	}



	//run compute on each layer

	for (int l = 1; l < layers.size(); l++) {
		layers[l].forwardCompute();
	}


	//return results

	std::vector<double> results = std::vector<double>();

	for (int i = 0; i < layers[layers.size() - 1].getNodes().size(); i++) {
		results.push_back(layers[layers.size() - 1].getNodes()[i].getValue());
	}

	return results;
}

double NeuralNetwork::calculateCurrentLoss() {
	double total = 0;

	std::vector<double> currentInputRow;
	std::vector<double> currentOutputRow;

	std::vector<double> outputFromCompute;
	
	for(int r = 0; r < getTrainingInputs().size(); r++) {

		currentInputRow = trainingInputs[r];
		currentOutputRow = trainingOutputs[r];

		outputFromCompute = forwardCompute(currentInputRow);

		//compare results

		for (int i = 0; i < getTrainingOutputs().size(); i++) {
			total += Helper::calculateLoss(outputFromCompute[i], getTrainingOutputs()[r][i]);
		}

		
	}


	return total / currentInputRow.size();
}

void NeuralNetwork::gradientDescentTraining(double targetLoss, int iterations, double lowerRandomizationBound, double upperRandomizationBound, int numberOfSteps, double stepSize) {

}

void NeuralNetwork::gradientDescentTraining(double targetLoss, int iterations) {

}

void NeuralNetwork::gradientDescentTraining(int iterations) {


}


int main() {
	return 0;
}
