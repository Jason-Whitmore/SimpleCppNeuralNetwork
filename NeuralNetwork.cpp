#include "NeuralNetwork.h"




NeuralNetwork::NeuralNetwork(int inputs, int outputs, std::vector<int> layerHeights){
	numWeights = 0;
	numBiases = 0;
	
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

	l.push_back(2);
	l.push_back(2);
	l.push_back(2);

	NeuralNetwork n = NeuralNetwork(1,1,l);
	std::vector<double> input = std::vector<double>();

	input.push_back(2);

	std::cout << n.runNetwork(input).at(0) << std::endl;

	while(true);

}
