#include "NeuralNetwork.h"
NeuralNetwork::NeuralNetwork(int numInputs, int numOutputs, int layerCount, int nodesPerLayer) {
	srand(NULL);
	//create layers
	for (int l = 0; l < layerCount + 2; l++) {
		layers.push_back(NodeLayer());
	}

	//nodes for first layer
	for (int i = 0; i < numInputs; i++) {
		layers[0].addNode(RELUNode());
		nodeCount++;
	}


	//create the nodes for the sandwhich layer
	for (int l = 1; l < layerCount + 1; l++) {

		for (int n = 0; n < nodesPerLayer; n++) {
			layers[l].addNode(RELUNode());
			nodeCount++;
		}

	}



	//nodes for last layer
	for (int i = 0; i < numOutputs; i++) {
		layers[layers.size() - 1].addNode(RELUNode());
		nodeCount++;
	}
	//add connections to the layers
	//use pointers here

	
	for (int l = 1; l < layers.size(); l++) {
		for (int s = 0; s < layers[l-1].getNodes().size(); s++) {
			for (int d = 0; d < layers[l].getNodes().size(); d++) {
				Connection* c = new Connection();
				//std::cout << &c << std::endl;
				//maybe use references instead? <-----fixed the problem, for now atleast.
				layers[l-1].getNodes()[s].addOutput(c);

				layers[l].getNodes()[d].addInput(c);

				
				connectionCount++;
				
			}
		}
	}

	
	randomizeAllVariables(-10,10);

}


NeuralNetwork::~NeuralNetwork(){
	
}




Node& NeuralNetwork::pickRandomNode() {
	int nodeNumber = Helper::randomNumber(0, getNodeCount());
	
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

Connection* NeuralNetwork::pickRandomConnection() {
	int connectionNumber = Helper::randomNumber(0, getConnectionCount());
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

unsigned long long NeuralNetwork::getNodeCount() {
	return nodeCount;
}

unsigned long long NeuralNetwork::getConnectionCount() {
	return connectionCount;
}

std::vector<std::vector<double>>& NeuralNetwork::getTrainingInputs() {
	return trainingInputs;
}

std::vector<std::vector<double>>& NeuralNetwork::getTrainingOutputs() {
	return trainingOutputs;
}

void NeuralNetwork::setTrainingInputs(std::vector<std::vector<double>> i) {
	trainingInputs = i;
}

void NeuralNetwork::setTrainingInputs(std::string fileName, std::string entrySeparator, char pointSeperator) {
	std::ifstream file(fileName + ".csv");

	std::string s = "";
	std::string line = "";
	std::string temp = "";


	int separatorIndex;
	int lineSeparator;
	int index = 0;

	std::vector<double> dp = std::vector<double>();

	std::stringstream b;
	b << file.rdbuf();

	s = b.str();

	while (s.size() > 0) {
		
		line = s.substr(0, s.find(pointSeperator));
		lineSeparator = s.find_first_of(pointSeperator);
		s = s.substr(lineSeparator + 1);
		

		dp = Helper::parseLineDouble(line, entrySeparator);
		getTrainingInputs().push_back(dp);
		dp.clear();
	}


}

void NeuralNetwork::setTrainingOutputs(std::vector<std::vector<double>> o) {
	trainingOutputs = o;
}

void NeuralNetwork::setTrainingOutputs(std::string fileName, std::string entrySeparator, char pointSeperator) {
	std::ifstream file(fileName + ".csv");

	std::string s = "";
	std::string line = "";
	std::string temp = "";


	int separatorIndex;
	int lineSeparator;
	int index = 0;

	std::vector<double> dp = std::vector<double>();

	std::stringstream b;
	b << file.rdbuf();

	s = b.str();

	while (s.size() > 0) {

		line = s.substr(0, s.find(pointSeperator));
		lineSeparator = s.find_first_of(pointSeperator);
		s = s.substr(lineSeparator + 1);


		dp = Helper::parseLineDouble(line, entrySeparator);
		getTrainingOutputs().push_back(dp);
		dp.clear();
	}

}

std::vector<double> NeuralNetwork::forwardCompute(std::vector<double> inputs) {
	if (inputs.size() != layers[0].getNodes().size()) {
		//exception here
	}
	//move inputs to the first layer of nodes

	for (int i = 0; i < layers[0].getNodes().size(); i++) {
		layers[0].getNodes()[i].setValue(inputs[i]);

		for (int x = 0; x < layers[0].getNodes()[i].getOutputs().size(); x++) {
			layers[0].getNodes()[i].getOutputs()[x]->setValue(inputs[i]);
		}
	}



	//run compute on each layer

	for (int l = 1; l < layers.size(); l++) {
		//std::cout << "Compute called on layer " << l << std::endl;
		layers[l].forwardCompute();
	}


	//return results

	std::vector<double> results = std::vector<double>();

	for (int i = 0; i < layers[layers.size() - 1].getNodes().size(); i++) {
		results.push_back(layers[layers.size() - 1].getNodes()[i].getValue());
	}



	return results;
}

//probably the issue
double NeuralNetwork::calculateCurrentLoss() {
	double total = 0;

	std::vector<double> currentInputRow;
	std::vector<double> currentOutputRow;

	std::vector<double> outputFromCompute;
	
	double a = 0;
	double b;
	double c;

	double loss = 0;

	
	//test each row
	

		

		for (int dp = 0; dp < trainingInputs.size(); dp++) {

			outputFromCompute = forwardCompute(trainingInputs[dp]);

			for (int c = 0; c < trainingOutputs[0].size(); c++) {

				a = outputFromCompute[c];
				b = trainingOutputs[dp][c];

				loss += Helper::calculateLoss(outputFromCompute[c],trainingOutputs[dp][c]);
			}
		}

	

		return loss / trainingInputs.size();
}

void NeuralNetwork::gradientDescentTraining(double targetLoss, int iterations, double lowerRandomizationBound, double upperRandomizationBound, double passMultiple, int numberOfSteps, double stepSize, bool printInfo) {
	int improvements = 0;

	double bestLoss = 99999999999;
	double currentLoss = 9999999999999;

	double currentStepSize;

	for (int i = 0; i < iterations && bestLoss > targetLoss; i++) {
		randomizeAllVariables(lowerRandomizationBound, upperRandomizationBound);

		for (unsigned long long pass = 0; pass < (getNodeCount() + getConnectionCount()) * passMultiple; pass++) {
			currentStepSize = stepSize * (1 - (pass / ((getNodeCount() + getConnectionCount()) * passMultiple)));

			
			if (Helper::randomNumber(0.0,1.0) < ((double)getNodeCount() / (getConnectionCount() + getNodeCount()))) {
				optimizeBias(pickRandomNode(), numberOfSteps, currentStepSize);
			} else {
				optimizeWeight(pickRandomConnection(), numberOfSteps, currentStepSize);
			}
		}
		//dynamic stepsize thing: stepSize * (1 -(pass / ((double)(nodeCount() + connectionCount()) * 3.0)))

		currentLoss = calculateCurrentLoss();

		if (currentLoss < bestLoss) {
			improvements++;
			bestLoss = currentLoss;
			saveBiases();
			saveWeights();
		}

		//update info

		if (printInfo) {
			system("CLS");
			std::cout << "Best Loss: " << std::to_string(bestLoss) << std::endl;
			std::cout << "Current Loss: " + std::to_string(currentLoss) << std::endl;
			std::cout << "Number of loss improvements: " + std::to_string(improvements) << std::endl;
			std::cout << "Progress: " << (int)((i / ((double)iterations)) * 100) << "%" << std::endl;
		}
		
	}


	loadBiases();
	loadWeights();

}

void NeuralNetwork::gradientDescentTraining(double targetLoss, int iterations) {
	gradientDescentTraining(targetLoss, iterations, -50, 50, 2, 5, 10, true);
}

void NeuralNetwork::gradientDescentTraining(int iterations) {
	gradientDescentTraining(0, iterations, -50, 50, 2, 5, 10, true);
}



void NeuralNetwork::loadWeights() {
	std::ifstream file ("weights.txt");
	std::string s = "";
	std::string l = "";

	

	if (file.is_open()) {
		while (file.good()) {
			std::getline(file, l);

			s += l;
		}
	} else {
		std::cout << "error";
	}

	//file >> s;

	//std::getline(file, s);

	std::vector<double> w = Helper::parseLineDouble(s, " ");

	
	for (unsigned long long i = 0; i < getConnectionCount(); i++) {
		getConnection(i)->setWeight(w[i]);
	}
}

void NeuralNetwork::loadBiases() {
	std::ifstream file("biases.txt");
	std::string s = "";
	std::string l = "";



	if (file.is_open()) {
		while (file.good()) {
			std::getline(file, l);

			s += l;
		}
	} else {
		std::cout << "error";
	}

	//file >> s;

	//std::getline(file, s);

	std::vector<double> b = Helper::parseLineDouble(s, " ");

	
	
	for (unsigned long long i = 0; i < getNodeCount(); i++) {
		getNode(i).setBias(b[i]);
	}
}

void NeuralNetwork::saveBiases() {
	std::string s = "";



	for (unsigned long long i = 0; i < getNodeCount(); i++) {
		s += std::to_string(getNode(i).getBias());
		s += " ";
	}

	std::ofstream file(biasSaveLocation);
	if (!file.is_open()) {
		std::cout << "error";
		std::cin >> s;
	}
	file << s;

	file.close();

}

void NeuralNetwork::saveWeights() {
	std::string s = "";

	for (unsigned long long i = 0; i < getConnectionCount(); i++) {
		s += std::to_string(getConnection(i)->getWeight());
		s += " ";
	}


	std::ofstream file (weightSaveLocation);
	if (!file.is_open()) {
		std::cout << "error";
		std::cin >> s;
	}
	file << s;
	
	file.close();
}

std::string NeuralNetwork::getBiasSaveLocation() {
	return biasSaveLocation;
}

std::string NeuralNetwork::getWeightSaveLocation() {
	return weightSaveLocation;
}

void NeuralNetwork::setBiasSaveLocation(std::string location) {
	biasSaveLocation = location;
}

void NeuralNetwork::setWeightSaveLocation(std::string location) {
	weightSaveLocation = location;
}





void NeuralNetwork::optimizeBias(Node n, int steps, double stepSize) {

	double oldLoss = 0;
	double newLoss = 0;

	double currentStep = stepSize;
	double b = 0;
	double a = 0;
	//values not changing
	for (int i = 0; i < steps; i++) {
		oldLoss = calculateCurrentLoss();

		b = n.getBias();
		n.setBias(n.getBias() + currentStep);
		a = n.getBias();

		newLoss = calculateCurrentLoss();

		if (oldLoss < newLoss) {
			currentStep *= -.5;
		}
		//std::cout << std::to_string(calculateCurrentLoss()) << std::endl;
	}


}

void NeuralNetwork::optimizeWeight(Connection* c, int steps, double stepSize) {


	double oldLoss = 0;
	double newLoss = 0;

	double currentStep = stepSize;

	for (int i = 0; i < steps; i++) {
		oldLoss = calculateCurrentLoss();
		c->setWeight(c->getWeight() + currentStep);
		newLoss = calculateCurrentLoss();

		if (oldLoss < newLoss) {
			currentStep *= -.5;
		}

	}



}

void NeuralNetwork::randomizeAllVariables(double min, double max) {
	for (int n = 0; n < getNodeCount(); n++) {
		getNode(n).setBias(Helper::randomNumber(min,max));
	}

	for (int c = 0; c < getConnectionCount(); c++) {
		getConnection(c)->setWeight(Helper::randomNumber(min, max));
	}
}

Node& NeuralNetwork::getNode(unsigned long long index) {

	

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

Connection* NeuralNetwork::getConnection(unsigned long long index) {
	
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

double NeuralNetwork::extractDoubleFromString(std::string s) {
	std::string d = "";

	int spaceIndex = s.find(" ");

	d = s.substr(0, spaceIndex - 1);
	s = s.substr(spaceIndex);

	return std::stoi(d);
}


void NeuralNetwork::testMethod() {
	


	for (int i = 0; i < getNodeCount(); i++) {
		getNode(i).setBias(i);
	}

	for (int i = 0; i < getConnectionCount(); i++) {
		getConnection(i)->setWeight(i);
	}

	saveBiases();
	saveWeights();
	std::cout << "Loss with trained variables: " << calculateCurrentLoss() << std::endl;

	debugBiases();
	debugWeights();

	randomizeAllVariables(-10, 10);
	std::cout << "Loss with random variables: " << calculateCurrentLoss() << std::endl;

	debugBiases();
	debugWeights();


	loadBiases();
	loadWeights();
	std::cout << "Loss with trained variables: " << calculateCurrentLoss() << std::endl;

	debugBiases();
	debugWeights();

}

void NeuralNetwork::debugWeights() {
	for (unsigned long long i = 0; i < getConnectionCount(); i++) {
		std::cout << "Connection : " << i << " Weight: " << getConnection(i)->getWeight() << std::endl;
	}
	
}

void NeuralNetwork::debugBiases() {
	for (unsigned long long i = 0; i < getNodeCount(); i++) {
		std::cout << "Node : " << i << " Bias: " << getNode(i).getBias() << std::endl;
	}
}






int main() {
	
	NeuralNetwork n = NeuralNetwork(1, 1, 1, 5);
	
	//n.saveWeights();
	//n.saveBiases();

	//n.loadBiases();
	//n.loadWeights();
	

	std::vector<double> test = std::vector<double>();
	//std::string t = "5,4,3,2,1";

	//test = Helper::parseLineDouble(t, ",");

	n.setWeightSaveLocation("c:/Users/Jason/Source/Repos/SimpleCppNeuralNetwork/SimpleNeuralNetwork/SimpleNeuralNetwork/weights.txt");
	n.setBiasSaveLocation("c:/Users/Jason/Source/Repos/SimpleCppNeuralNetwork/SimpleNeuralNetwork/SimpleNeuralNetwork/bias.txt");

	
	n.setTrainingInputs(Helper::csvToTable("c:/Users/Jason/Source/Repos/SimpleCppNeuralNetwork/SimpleNeuralNetwork/SimpleNeuralNetwork/TInputs.csv", "\n", ",", 1, 20, 0, 0));
	n.setTrainingOutputs(Helper::csvToTable("c:/Users/Jason/Source/Repos/SimpleCppNeuralNetwork/SimpleNeuralNetwork/SimpleNeuralNetwork/TOutputs.csv", "\n", ",", 1, 20, 0, 0));
	

	n.gradientDescentTraining(0, 100, -100, 100, 2, 10, 10, true);

	return 0;
}
