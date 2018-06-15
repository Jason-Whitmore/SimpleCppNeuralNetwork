#pragma once
class NodeLayer {
	public:
	NodeLayer();
	~NodeLayer();

	private:

	double* inputArray;

	double* inputWeights;

	double* outputs;

	double* outputBiases;

};

