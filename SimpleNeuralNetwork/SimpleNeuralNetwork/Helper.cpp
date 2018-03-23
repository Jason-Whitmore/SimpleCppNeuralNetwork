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

	return std::abs(a - b) / std::fmax(std::abs(a),std::abs(b));

}

double Helper::activationFunctionRELU(double sum, double bias) {
	double input = sum + bias;

	if (input >= 0) {
		return input;
	} else {
		return input * 0.001;
	}
}





std::vector<double> Helper::parseLineDouble(std::string target, std::string entrySeparator) {

	std::vector<std::string> aux = parseLineString(target, entrySeparator);

	std::vector<double> r = std::vector<double>();

	for (unsigned long long i = 0; i < aux.size(); i++) {
		r.push_back(std::stod(aux[i]));
	}

	return r;
}

std::vector<std::string> Helper::parseLineString(std::string target, std::string entrySeparator) {
	std::vector<std::string> r = std::vector<std::string>();

	int separatorIndex = target.find_first_of(entrySeparator);

	std::string entry = "";
	while (target.size() > 0) {
		//do the parse
		separatorIndex = target.find_first_of(entrySeparator);

		//separator is still in the string
		if (separatorIndex >= 0) {
			entry = target.substr(0, separatorIndex);

			r.push_back(entry);

			target = target.substr(separatorIndex + 1);
			
		} else {
			//no more separator, nearing the end of the list
			if (target.length() != 0) {
				r.push_back(target);
			}

			target = "";
		}

		separatorIndex = target.find_first_of(entrySeparator);
	}


	return r;
}



std::vector<std::vector<std::string>> Helper::csvToVector(std::string filePath, std::string rowSeparator, std::string entrySeparator) {
	std::string s = "";

	//put the contents of the file into the string

	std::ifstream file(filePath);

	std::stringstream b;
	b << file.rdbuf();

	s = b.str();

	//set up the variables
	std::vector<std::vector<std::string>> r = std::vector<std::vector<std::string>>();

	std::vector<std::string> rows = parseLineString(s, rowSeparator);

	std::vector<std::string> entries = std::vector<std::string>();

	//parse the rows and entries
	for (unsigned long long row = 0; row < rows.size(); row++) {
		
		entries = parseLineString(rows[row], entrySeparator);
		r.push_back(entries);
		entries.clear();

	}

	return r;
}


