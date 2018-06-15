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
