#pragma once
#include "Helper.h"
class NodeLayer {
	public:
	NodeLayer(int numInputs, int numNodes);
	~NodeLayer();

	void dotProductAndApplyBias();
	void dotProduct();

	void setInputArray(double a[]);
	void setInputArray(std::vector<double> v);

	double* getOutputArray();
	std::vector<double> getOutputVector();

	private:

	double* inputArray;

	double** inputWeights;

	double* layerOutputs;

	double* outputBiases;

	int numInputs;

	int numOutputs;

};

