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
	
	for (int i = 0; i < iterations; i++) {

		for (unsigned long long pass = 0; pass < (nodeCount() + connectionCount()) * 3; pass++) {
			if (Helper::randomNumber(0.0,1.0) < ((double)nodeCount() / (connectionCount() + nodeCount()))) {
				optimizeBias(pickRandomNode(), numberOfSteps, stepSize * (1 - (pass / ((double)(nodeCount() + connectionCount()) * 3))));
			} else {
				optimizeWeight(pickRandomConnection(), numberOfSteps, stepSize * (1 -(pass / ((double)(nodeCount() + connectionCount()) * 3))));
			}
		}
	
	}


}

void NeuralNetwork::gradientDescentTraining(double targetLoss, int iterations) {

}

void NeuralNetwork::gradientDescentTraining(int iterations) {


}

void NeuralNetwork::optimizeBias(Node n, int steps, double stepSize) {

	double oldLoss = 0;
	double newLoss = 0;

	double currentStep = stepSize;

	for (int i = 0; i < steps; i++) {
		oldLoss = calculateCurrentLoss();
		n.setBias(n.getBias() + currentStep);
		newLoss = calculateCurrentLoss();

		if (oldLoss < newLoss) {
			currentStep *= -.5;
		}

	}


}

void NeuralNetwork::optimizeWeight(Connection c, int steps, double stepSize) {


	double oldLoss = 0;
	double newLoss = 0;

	double currentStep = stepSize;

	for (int i = 0; i < steps; i++) {
		oldLoss = calculateCurrentLoss();
		c.setWeight(c.getWeight() + currentStep);
		newLoss = calculateCurrentLoss();

		if (oldLoss < newLoss) {
			currentStep *= -.5;
		}

	}



}

void NeuralNetwork::randomizeAllVariables() {
	for (int n = 0; n < nodeCount(); n++) {
		
	}
}

Node NeuralNetwork::getNode(unsigned long long index) {

	

	unsigned long long currentNode = 0;
	for (int l = 0; l < layers.size(); l++) {

		for (int n = 0; n < layers[l].getNodes().size(); n++) {
			if (currentNode != index) {
				currentNode++;
			} else {
				return layers[l].getNodes()[n];
			}
		}

	}


}

Connection NeuralNetwork::getConnection(unsigned long long index) {
	
	unsigned long long currentConnection = 0;

	for (int l = 1; l < layers.size(); l++) {

		for (int n = 0; n < layers[l].getNodes().size(); n++) {
			for (int c = 0; c < layers[l].getNodes()[n].getInputs().size(); c++) {
				if (currentConnection != index) {
					currentConnection++;
				} else {
					return layers[l].getNodes()[n].getInputs()[c];
				}
			}
		}
	}
}


int main() {
	return 0;
}
