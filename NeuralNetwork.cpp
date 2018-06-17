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


void NeuralNetwork::hyperparameterOptimization(int maxNodes, double randMin, double randMax) {
	
}

void NeuralNetwork::trainNetwork(double targetLoss, int maxIterations, int numOfSteps, double numPassesScalar, double stepSize, double randMin, double randMax, bool displayStats) {
	std::vector<NodeLayer>* bestLayers = new std::vector<NodeLayer>(*layers);
	double currentLoss = calculateCurrentLoss();
	double bestLoss = currentLoss;

	int improvements = 0;

	for(int i = 0; i < maxIterations && bestLoss > targetLoss; i++) {
		randomizeVariables(randMin, randMax);


		for (int pass = 0; pass < (numBiases + numWeights) * numPassesScalar; pass++) {
			optimizeRandomVariable(numOfSteps, stepSize, randMin, randMax);
		}

		currentLoss = calculateCurrentLoss();

		if (bestLoss > currentLoss) {
			bestLayers = new std::vector<NodeLayer>(*layers);
			improvements++;
		}

		if (displayStats && i % 100 == 0) {
			system("CLS");
			std::cout << "Iteration: " << i << "/" << maxIterations << std::endl;
			std::cout << "BestLoss: " << bestLoss << std::endl;
			std::cout << "Number of Improvements to loss: " << improvements << std::endl;
			std::cout << "Current loss: " << currentLoss << std::endl;
		}


	}

	layers = bestLayers;


}

void NeuralNetwork::optimizeRandomVariable(int numOfSteps, double stepSize, double randMin, double randMax) {
	//need to find proportion of biases to totalVariables
	double biasesToTotalVariables = numBiases / (numBiases + numWeights);

	if (Helper::randomDouble(0,1) > biasesToTotalVariables) {
		//pick a weight to optimize
		int weightIndex = Helper::randomInt(0, numWeights - 1);

		int currentWeightIndex = 0;
		int currentLayerIndex = 0;

		//find layer that contains the right index
		while (currentWeightIndex + layers->at(currentLayerIndex).getNumWeights() < weightIndex) {
			currentWeightIndex += layers->at(currentLayerIndex).getNumWeights();

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
		int biasIndex = Helper::randomInt(0, numBiases - 1);

		int currentBiasIndex = 0;
		int currentLayerIndex = 0;

		//find layer that contains the right index
		while (currentBiasIndex + layers->at(currentLayerIndex).getNumBiases() < biasIndex) {
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

void NeuralNetwork::setTrainingInputs(Data inputs) {
	trainingInputs = inputs;
}

void NeuralNetwork::setTrainingOutputs(Data outputs) {
	trainingOutputs = outputs;
}



double NeuralNetwork::calculateCurrentLoss() {
	double dataPointCount = 0;
	double totalLoss = 0;

	std::vector<double> networkOutputs = std::vector<double>();

	for (int r = 0; r < trainingOutputs.getNumRows(); r++) {
		//get the network's output
		networkOutputs = runNetwork(Helper::arrayToVector(trainingInputs.getRow(r), trainingInputs.getNumCols()));

		//caclulate loss and add to sum
		for (int c = 0; c < trainingOutputs.getNumCols(); c++) {
			
			totalLoss += Helper::calculateLoss(trainingOutputs.getIndex(r,c), networkOutputs[c]);
			dataPointCount++;
		}
	}
	return totalLoss / dataPointCount;
}

int main() {
	std::vector<int> l = std::vector<int>();

	l.push_back(3);

	NeuralNetwork n = NeuralNetwork(1,1,l);
	//train network

	Data trainingInputs = Data(9, 1);
	Data trainingOutputs = Data(9, 1);

	for (double i = -4; i < 5; i++) {
		trainingInputs.setIndex(i + 4,0,i);
		trainingOutputs.setIndex(i + 4, 0, std::pow(i,2));
	}

	n.setTrainingInputs(trainingInputs);
	n.setTrainingOutputs(trainingOutputs);

	n.trainNetwork(0,1000000, 4, 3, 1, -5, 5, true);

	std::vector<double> input = std::vector<double>();

	input.push_back(2);

	std::cout << n.runNetwork(input).at(0) << std::endl;

	while(true);

}
