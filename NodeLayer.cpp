#include "NodeLayer.h"



NodeLayer::NodeLayer(int inputs, int outputs) {
	inputArray = new double[inputs];

	//create weight 2d array (a bit complicated)
	inputWeights = new double*[outputs];

	for (int i = 0; i < outputs; i++) {
		inputWeights[i] = new double[inputs];
	}

	//remove this later
	for (int r = 0; r < outputs; r++) {
		for (int c = 0; c < inputs; c++) {
			inputWeights[r][c] = 1;
		}
	}


	layerOutputs = new double[outputs];

	outputBiases = new double[outputs];

	for (int i = 0; i < outputs; i++) {
		layerOutputs[i] = 0;
		outputBiases[i] = 0;
	}

	numOutputs = outputs;
	numInputs = inputs;

}


NodeLayer::~NodeLayer() {
	

}



void NodeLayer::dotProductAndApplyBias() {

	//perform dot products
	for (int i = 0; i < numOutputs; i++) {
		layerOutputs[i] = Helper::dotProduct(inputArray, inputWeights[i], numInputs);
	}

	//apply biases
	for (int i = 0; i < numOutputs; i++) {
		layerOutputs[i] = Helper::RELUFunction(layerOutputs[i], outputBiases[i]);
	}

}

void NodeLayer::dotProduct() {

	//perform dot products
	for (int i = 0; i < numOutputs; i++) {
		layerOutputs[i] = Helper::dotProduct(inputArray, inputWeights[i], numInputs);
	}

}

void NodeLayer::setInputArray(double a[]) {
	for (int i = 0; i < numInputs; i++) {
		inputArray[i] = a[i];
	}
}

void NodeLayer::setInputArray(std::vector<double> v) {
	for (int i = 0; i < numInputs; i++) {
		inputArray[i] = v[i];
	}
}

double* NodeLayer::getOutputArray() {
	return layerOutputs;
}

std::vector<double> NodeLayer::getOutputVector() {
	std::vector<double> r = std::vector<double>();

	for (int i = 0; i < numOutputs; i++) {
		r.push_back(layerOutputs[i]);
	}

	return r;
}
