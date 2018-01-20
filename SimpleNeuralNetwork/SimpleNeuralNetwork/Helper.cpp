#include "Helper.h"
#include <random>

Helper::Helper() {
}


Helper::~Helper() {
}

int Helper::randomNumber(int a, int b) {
	return a + (int)(rand() % (b - a));
}

double Helper::randomNumber(double a, double b) {
	return a + (rand() / ((double)RAND_MAX)) * (b-a);
}

double Helper::calculateLoss(double value, double target) {
	double similarity;

	if (value > target) {
		if (value == 0) {
			return 0;
		}

		similarity = target / value;
	} else {
		if (target == 0) {
			return 0;
		}

		similarity = value / target;
	}

	return 1 - similarity;
}

double Helper::activationFunctionRELU(double sum, double bias) {
	double input = sum + bias;

	if (input >= 0) {
		return input;
	} else {
		return input * 0.1;
	}
}


