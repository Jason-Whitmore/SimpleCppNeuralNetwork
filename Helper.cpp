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
	
	double num;
	for (int i = 0; i < arraySize; i++) {
		num = array[i];
		r.push_back(array[i]);
	}

	return r;
}

double Helper::randomDouble(double min, double max) {
	double scalar = (double)rand() / RAND_MAX;

	return min + (scalar * (max - min));
}

int Helper::randomInt(int min, int max) {
	
	return min + (rand() % (max - min));
}

std::vector<std::vector<int>> Helper::getSetOfLayerConfigs(int numNodes) {

	std::vector<std::vector<int>> r = std::vector<std::vector<int>>();


	std::vector<int> currentConfig;
	//while all configs have not been found yet....
	while (r.size() < std::pow(2, numNodes - 1)) {
		currentConfig = generateConfig(numNodes);

		//if unique, add to return vector
		if (!contains(currentConfig, r)) {
			r.push_back(currentConfig);
		}
	}

	return r;
}

std::vector<int> Helper::generateConfig(int numNodes) {
	std::vector<int> r = std::vector<int>();
	
	int nodesLeft = numNodes;
	int rand;

	while(nodesLeft >= 1) {
		if (nodesLeft == 1) {
			r.push_back(1);
			return r;
		}
		

		rand = Helper::randomInt(1, nodesLeft + 1);

		nodesLeft -= rand;
		
		r.push_back(rand);

	}


	return r;
}

bool Helper::contains(std::vector<int> v, std::vector<std::vector<int>> setOfVectors) {


	for (int i = 0; i < setOfVectors.size(); i++) {
		if (setOfVectors[i] == v) {
			return true;
		}
	}

	return false;
}

std::vector<std::string> Helper::split(std::string s, std::string splitter) {
	std::vector<std::string> r = std::vector<std::string>();

	std::string copy = s;
	
	while (Helper::contains(copy, splitter)) {
		r.push_back(copy.substr(0, copy.find_first_of(splitter)));
		copy = copy.substr(copy.find_first_of(splitter));
	}

	r.push_back(copy);


	return r;
	
}

bool Helper::contains(std::string s, std::string targetString) {
	int targetLength = targetString.length();

	int currentPosition = 0;

	while (currentPosition + targetLength <= s.length()) {
		if (s.substr(currentPosition,targetLength) == targetString) {
			return true;
		}
		currentPosition++;
	}

	return false;
}
