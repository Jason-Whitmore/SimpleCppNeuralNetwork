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



double Helper::calculateSimilarity(double a, double b) {
	//return Math.Abs(a - b);


	if (a > b) {
		if (a == 0) {
			return 0;
		}

		return b / a;
	} else {
		if (b == 0) {
			return 0;
		}

		return a / b;
	}
}




double Helper::calculateLoss(double a, double b) {

	return 1 - calculateSimilarity(a,b);

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
		separatorIndex = target.find_first_of(entrySeparator);
		
		
		if (separatorIndex >= 0) {
			number = target.substr(0, separatorIndex);

			r.push_back(std::stod(number));

			target = target.substr(separatorIndex + 1);
		} else {
			r.push_back(std::stod(target));
			target = "";
		}
		
		



		separatorIndex = target.find_first_of(entrySeparator);
	}


	return r;
}


