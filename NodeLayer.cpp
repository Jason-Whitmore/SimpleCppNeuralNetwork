#include "NodeLayer.h"



NodeLayer::NodeLayer(int numInputs, int numOutputs) {
	inputArray = new double[numInputs];

	inputWeights = new double*[numInputs];

	outputs = new double[numOutputs];

	outputBiases = new double[numOutputs];


}


NodeLayer::~NodeLayer() {
}

void NodeLayer::multiplyAndApplyBias() {


}
