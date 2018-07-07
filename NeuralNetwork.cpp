#include "NeuralNetwork.h"




NeuralNetwork::NeuralNetwork(int inputs, int outputs, std::vector<int> layerHeights){
	numWeights = 0;
	numBiases = 0;
	
	srand(NULL);
	//loop and create the layers properly

	//create first layer
	//layers->push_back(new NodeLayer(inputs, layerHeights[0]));

	layers = new std::vector<NodeLayer>();

	layerHeights.insert(layerHeights.begin(), inputs);
	layerHeights.push_back(outputs);

	layers->push_back(NodeLayer(0, inputs));
	numBiases += inputs;
	

	//loop through to create layers
	for (int i = 1; i < layerHeights.size(); i++) {
		layers->push_back(NodeLayer(layerHeights[i-1], layerHeights[i]));
		numBiases += layerHeights[i];
		numWeights += layerHeights[i] * layerHeights[i-1];
	}

	numInputs = inputs;
	numOutputs = outputs;
}

NeuralNetwork::~NeuralNetwork() {
	delete layers;
}



std::vector<double> NeuralNetwork::runNetwork(std::vector<double> inputs) {

	layers->at(1).setInputArray(inputs);

	for (int i = 1; i < layers->size() - 1; i++) {
		layers->at(i).dotProductAndApplyBias();
		layers->at(i+1).setInputArray(layers->at(i).getOutputArray());
	}
	layers->at(layers->size()-1).dotProduct();

	return layers->at(layers->size()-1).getOutputVector();
}

std::vector<double> NeuralNetwork::runNetwork(double * inputs) {
	layers->at(1).setInputArray(inputs);

	for (int i = 1; i < layers->size() - 1; i++) {
		layers->at(i).dotProductAndApplyBias();
		layers->at(i + 1).setInputArray(layers->at(i).getOutputArray());
	}
	layers->at(layers->size() - 1).dotProduct();

	return layers->at(layers->size() - 1).getOutputVector();
}





void NeuralNetwork::trainNetwork(double targetLoss, int maxIterations, int numOfSteps, double numPassesScalar, double stepSize, double randMin, double randMax, bool displayStats) {
	std::vector<NodeLayer>* startLayers = layers;
	std::vector<NodeLayer>* bestLayers = layers;
	double currentLoss = calculateCurrentLoss();
	double bestLoss = currentLoss;

	int improvements = 0;

	for(int i = 0; i < maxIterations && bestLoss > targetLoss; i++) {
		randomizeVariables(randMin, randMax);


		for (int pass = 0; pass < ((numBiases - numInputs) + numWeights) * numPassesScalar; pass++) {
			optimizeRandomVariable(numOfSteps, stepSize, randMin, randMax);
			std::cout << "Completed pass " << pass <<". New Loss: " << calculateCurrentLoss() << std::endl;
		}

		currentLoss = calculateCurrentLoss();

		if (bestLoss > currentLoss) {
			
			//bestLayers = new std::vector<NodeLayer>(*layers);
			bestLoss = currentLoss;
			//layers = startLayers;
			saveBiases();
			saveWeights();
			improvements++;
			//debugLayers();

		}

		if (displayStats && i % 10 == 0) {
			system("CLS");
			std::cout << "Iteration: " << i << "/" << maxIterations << std::endl;
			std::cout << "Best Loss: " << bestLoss << std::endl;
			std::cout << "Number of Improvements to loss: " << improvements << std::endl;
			std::cout << "Current Loss: " << currentLoss << std::endl;
			
		}
		

	}




	loadBiases("biases.txt");
	loadWeights("weights.txt");

	//debugLayers();
	return;
	//debugLayers();
}

void NeuralNetwork::optimizeRandomVariable(int numOfSteps, double stepSize, double randMin, double randMax) {
	//need to find proportion of biases to totalVariables
	double biasesToTotalVariables = ((double)numBiases - numInputs) / (numBiases - numInputs + numWeights);

	if (Helper::randomDouble(0,1) > biasesToTotalVariables) {
		//pick a weight to optimize
		int weightIndex = Helper::randomInt(0, numWeights);

		int currentWeightIndex = 0;
		int currentLayerIndex = 1;

		//find layer that contains the right index
		while (currentWeightIndex + layers->at(currentLayerIndex).getNumWeights() <= weightIndex) {
			currentWeightIndex += layers->at(currentLayerIndex).getNumWeights();
			currentLayerIndex++;
		}

		//layer index found
		//calculate the index for the weight within the current layer
		weightIndex = weightIndex - currentWeightIndex;

		double prevLoss;
		double currentDelta = stepSize;
		for (int i = 0; i < numOfSteps; i++) {
			prevLoss = calculateCurrentLoss();
			//make changes
			layers->at(currentLayerIndex).setWeight(weightIndex, layers->at(currentLayerIndex).getWeight(weightIndex) + currentDelta);

			//backtrack if loss got worse
			if (calculateCurrentLoss() > prevLoss) {
				currentDelta *= -0.5;
			}


		}
		



	} else {
		//pick a bias to optimize
		//pick a weight to optimize
		int biasIndex = Helper::randomInt(numInputs, numBiases);

		

		int currentBiasIndex = 0;
		int currentLayerIndex = 0;

		//find layer that contains the right index
		while (currentBiasIndex + layers->at(currentLayerIndex).getNumBiases() <= biasIndex) {
			currentBiasIndex += layers->at(currentLayerIndex).getNumBiases();

		}

		//layer index found
		//calculate the index for the weight within the current layer
		biasIndex = biasIndex - currentBiasIndex;

		double prevLoss;
		double currentDelta = stepSize;
		for (int i = 0; i < numOfSteps; i++) {
			prevLoss = calculateCurrentLoss();
			//make changes
			layers->at(currentLayerIndex).setBias(biasIndex, layers->at(currentLayerIndex).getBias(biasIndex) + currentDelta);

			//backtrack if loss got worse
			if (calculateCurrentLoss() > prevLoss) {
				currentDelta *= -0.5;
			}

		}

	}


}

void NeuralNetwork::randomizeVariables(double min, double max) {
	
	for (int i = 0; i < layers->size(); i++) {
		//set weights
		for (int w = 0; w < layers->at(i).getNumWeights(); w++) {
			layers->at(i).setWeight(w, Helper::randomDouble(min,max));
		}

		//set biases

		for (int b = 0; b < layers->at(i).getNumBiases(); b++) {
			layers->at(i).setBias(b, Helper::randomDouble(min,max));
		}

	}

}

void NeuralNetwork::setTrainingInputs(Data* inputs) {
	trainingInputs = inputs;
}

void NeuralNetwork::setTrainingOutputs(Data* outputs) {
	trainingOutputs = outputs;
}

Data * NeuralNetwork::getTrainingInputs() {
	return trainingInputs;
}

Data * NeuralNetwork::getTrainingOutputs() {
	return trainingOutputs;
}

void NeuralNetwork::debugLayers() {

	for (int i = 0; i < layers->size(); i++) {
		std::cout << "Layer " << i << std::endl;

		//incoming weights
		std::cout << "\tWeights: " << std::endl;

		for(int a = 0; a < layers->at(i).getNumWeights(); a++) {
			std::cout << "\t\tWeight " << a << ":" << layers->at(i).getWeight(a) << std::endl;
		}

		//incoming biases
		std::cout << "\tBiases: " << std::endl;

		for (int a = 0; a < layers->at(i).getNumBiases(); a++) {
			std::cout << "\t\tBias " << a << ":" << layers->at(i).getBias(a) << std::endl;
		}
	}
}

void NeuralNetwork::debugLayer(int layerNum) {
	std::cout << "Layer " << layerNum << std::endl;

	std::cout << "\tWeights: " << std::endl;

	for (int i = 0; i < layers->at(layerNum).getNumWeights(); i++) {
		std::cout << "\t\tWeight " << i << ":" << layers->at(layerNum).getWeight(i) << std::endl;
	}

	std::cout << "\tBiases: " << std::endl;

	for (int i = 0; i < layers->at(layerNum).getNumBiases(); i++) {
		std::cout << "\t\\tBiases " << i << ":" << layers->at(layerNum).getBias(i) << std::endl;
	}


}

void NeuralNetwork::saveWeights() {
	std::vector<double> weights = getAllWeights();

	std::string output = "";

	for (int i = 0; i < weights.size(); i++) {
		output += std::to_string(weights[i]);
		output += "\n";
	}

	std::ofstream file;
	file.open("weights.txt");

	file << output;

	file.close();

}

void NeuralNetwork::saveBiases() {
	std::vector<double> biases = getAllBiases();

	std::string output = "";

	for (int i = 0; i < biases.size(); i++) {
		output += std::to_string(biases[i]);
		output += "\n";
	}

	std::ofstream file;
	file.open("biases.txt");

	file << output;

	file.close();
}


std::vector<double> NeuralNetwork::getAllBiases() {
	std::vector<double> r = std::vector<double>();

	for (int i = 0; i < numBiases; i++) {
		r.push_back(getBias(i));
	}


	return r;
}

std::vector<double> NeuralNetwork::getAllWeights() {
	std::vector<double> r = std::vector<double>();

	for (int i = 0; i < numWeights; i++) {
		r.push_back(getWeight(i));
	}


	return r;
}

void NeuralNetwork::loadBiases(std::string filePath) {
	std::string singleLine;
	std::ifstream file(filePath);

	double bias;
	unsigned int counter = 0;

	if (file.is_open()) {

		while (std::getline(file, singleLine)) {
			//interpret data
			bias = std::stod(singleLine);
			//place in correct spot
			setBias(counter, bias);
			counter++;
		}


	} else {
		//file could not be opened
	}
}

void NeuralNetwork::loadWeights(std::string filePath) {
	std::string singleLine;
	std::ifstream file(filePath);

	double weight;
	unsigned int counter = 0;

	if (file.is_open()) {

		while (std::getline(file, singleLine)) {
			//interpret data
			weight = std::stod(singleLine);
			//place in correct spot
			setWeight(counter, weight);
			counter++;
		}


	} else {
		//file could not be opened
	}
}

double NeuralNetwork::getBias(int index) {
	int biasIndex = index;

	int currentBiasIndex = 0;
	int currentLayerIndex = 0;

	//find layer that contains the right index
	while (currentBiasIndex + layers->at(currentLayerIndex).getNumBiases() <= biasIndex) {
		currentBiasIndex += layers->at(currentLayerIndex).getNumBiases();
		currentLayerIndex++;
	}

	//layer index found
	//calculate the index for the weight within the current layer
	biasIndex = biasIndex - currentBiasIndex;

	double result = layers->at(currentLayerIndex).getBias(biasIndex);
	

	return result;

}

double NeuralNetwork::getWeight(int index) {

	int weightIndex = index;

	int currentWeightIndex = 0;
	int currentLayerIndex = 1;

	//find layer that contains the right index
	while (currentWeightIndex + layers->at(currentLayerIndex).getNumWeights() <= weightIndex) {
		currentWeightIndex += layers->at(currentLayerIndex).getNumWeights();
		currentLayerIndex++;
	}

	//layer index found
	//calculate the index for the weight within the current layer
	weightIndex = weightIndex - currentWeightIndex;

	return layers->at(currentLayerIndex).getWeight(weightIndex);
}

void NeuralNetwork::setBias(int index, double value) {
	int biasIndex = index;

	int currentBiasIndex = 0;
	int currentLayerIndex = 0;

	//find layer that contains the right index
	while (currentBiasIndex + layers->at(currentLayerIndex).getNumBiases() <= biasIndex) {
		currentBiasIndex += layers->at(currentLayerIndex).getNumBiases();
		currentLayerIndex++;
	}

	//layer index found
	//calculate the index for the weight within the current layer
	biasIndex = biasIndex - currentBiasIndex;

	layers->at(currentLayerIndex).setBias(biasIndex, value);

}

void NeuralNetwork::setWeight(int index, double value) {

	int weightIndex = index;

	int currentWeightIndex = 0;
	int currentLayerIndex = 1;

	//find layer that contains the right index
	while (currentWeightIndex + layers->at(currentLayerIndex).getNumWeights() <= weightIndex) {
		currentWeightIndex += layers->at(currentLayerIndex).getNumWeights();
		currentLayerIndex++;
	}

	//layer index found
	//calculate the index for the weight within the current layer
	weightIndex = weightIndex - currentWeightIndex;

	return layers->at(currentLayerIndex).setWeight(weightIndex, value);
}



double NeuralNetwork::calculateCurrentLoss() {
	double dataPointCount = 0;
	double totalLoss = 0;
	double loss = 0;

	std::vector<double> networkOutputs = std::vector<double>();

	//bug somewhere here
	for (int r = 0; r < trainingOutputs->getNumRows(); r++) {
		//get the network's output
		double* dataRow = trainingInputs->getRow(r);
		networkOutputs = runNetwork(dataRow);
		//networkOutputs = runNetwork(Helper::arrayToVector(dataRow, trainingInputs->getNumCols()));
		delete dataRow;
		//caclulate loss and add to sum
		for (int c = 0; c < trainingOutputs->getNumCols(); c++) {

			loss += Helper::calculateLoss(trainingOutputs->getIndex(r,c), networkOutputs[c]) * (1.0 / trainingOutputs->getNumCols());
			dataPointCount++;
		}
	}
	return loss / trainingOutputs->getNumRows();
}

