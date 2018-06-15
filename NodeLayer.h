#pragma once
#include "Helper.h"
class NodeLayer {
	public:
	NodeLayer(int numInputs, int numNodes);
	~NodeLayer();

	void multiplyAndApplyBias();

	private:

	double* inputArray;

	double** inputWeights;

	double* outputs;

	double* outputBiases;

	int numInputs;

	int numOutputs;

};

