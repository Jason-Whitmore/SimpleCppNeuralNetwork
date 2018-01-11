#include "NeuralNetwork.h"


NeuralNetwork::NeuralNetwork(){
	
}

NeuralNetwork::NeuralNetwork(int numInputs, int numOutputs, int layerCount, int nodesPerLayer) {

	//create layers
	for (int l = 0; l < layerCount + 2; l++) {
		layers.push_back(NodeLayer());
	}

	//nodes for first layer
	for (int i = 0; i < numInputs; i++) {
		layers[0].addNode(RELUNode());
	}

	//nodes for last layer
	for (int i = 0; i < numOutputs; i++) {
		layers[layers.size() - 1].addNode(RELUNode());
	}




	//create the nodes for the sandwhich layer
	for (int l = 1; l < layerCount - 1; l++) {

		for (int n = 0; n < nodesPerLayer; n++) {

			layers[l].addNode(RELUNode());

		}

	}

	//add connections to the layers
	Connection c;
	for (int l = 1; l < layers.size(); l++) {
		for (int s = 0; s < layers[l-1].getNodes().size(); s++) {
			for (int d = 0; d < layers[l].getNodes().size(); d++) {
				c = Connection();
				layers[l-1].getNodes()[s].getOutputs().push_back(c);
				layers[l].getNodes()[d].getInputs().push_back(c);

			}
		}
	}



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

		for (int x = 0; x < layers[0].getNodes()[i].getOutputs().size(); x++) {
			layers[0].getNodes()[i].getOutputs()[x].setValue(inputs[i]);
		}
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
		randomizeAllVariables(lowerRandomizationBound, upperRandomizationBound);


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
	gradientDescentTraining(targetLoss, iterations, -50, 50, 10, 5);
}

void NeuralNetwork::gradientDescentTraining(int iterations) {
	gradientDescentTraining(0, iterations, -50, 50, 10, 5);
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

void NeuralNetwork::randomizeAllVariables(double min, double max) {
	for (int n = 0; n < nodeCount(); n++) {
		getNode(n).setBias(Helper::randomNumber(min,max));
	}

	for (int c = 0; c < connectionCount(); c++) {
		getConnection(c).setWeight(Helper::randomNumber(min,max));
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

std::vector<double> NeuralNetwork::getAllWeights() {
	return std::vector<double>();
}

std::vector<double> NeuralNetwork::getAllBiases() {
	return std::vector<double>();
}

void NeuralNetwork::setWeights(std::vector<double> w) {
}

void NeuralNetwork::setBiases(std::vector<double> b) {
}


int main() {
	return 0;
}
