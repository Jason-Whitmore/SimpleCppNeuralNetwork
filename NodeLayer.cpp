#include "NodeLayer.h"



NodeLayer::NodeLayer(int numInputs, int numOutputs) {
	inputArray = new double[numInputs];

	//create weight 2d array (a bit complicated)
	inputWeights = new double*[numInputs];

	for (int i = 0; i < numInputs; i++) {
		inputWeights[i] = new double[numOutputs];
	}



	outputs = new double[numOutputs];

	outputBiases = new double[numOutputs];

	numOutputs = numOutputs;
	numInputs = numInputs;

}


NodeLayer::~NodeLayer() {
	

}



void NodeLayer::multiplyAndApplyBias() {

	//perform dot products
	for (int i = 0; i < numOutputs; i++) {
		outputs[i] = Helper::dotProduct(inputArray, inputWeights[i], numInputs);
	}

	//apply biases
	for (int i = 0; i < numOutputs; i++) {
		outputs[i] = Helper::RELUFunction(outputs[i], outputBiases[i]);
	}

}
