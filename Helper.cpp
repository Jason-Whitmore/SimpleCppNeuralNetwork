#include "Helper.h"



Helper::Helper() {
}


Helper::~Helper() {
}

double Helper::dotProduct(double a[], double b[], int length) {

	double result = 0;
	for(int i = 0; i < length; i++) {
		result += a[i] * b[i];
	}

	return result;
}



double Helper::RELUFunction(double input, double bias) {
	double newInput = input + bias;

	if (newInput < 0) {
		return newInput * .01;
	} else {
		return newInput;
	}
}



double Helper::calculateLoss(double value1, double value2) {
	return (value1 - value2) * (value1 - value2);
}

std::vector<double> Helper::arrayToVector(double array[], int arraySize) {
	std::vector<double> r = std::vector<double>();

	for (int i = 0; i < arraySize; i++) {
		r.push_back(array[i]);
	}

	return r;
}
