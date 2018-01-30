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





std::vector<double> Helper::parseLine(std::string target, std::string entrySeparator) {
	std::vector<double> r = std::vector<double>();

	int separatorIndex = target.find_first_of(entrySeparator);

	std::string number = "";
	while (target.size() > 0) {
		//do the parse

		
		
		if (separatorIndex >= 0) {
			number = target.substr(0, separatorIndex);

			r.push_back(std::stod(number));

			target = target.substr(separatorIndex + 1);
		} else {
			r.push_back(std::stod(target));
		}
		
		



		separatorIndex = target.find_first_of(entrySeparator);
	}


	return r;
}


