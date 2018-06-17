#pragma once
#include <vector>
#include <random>
class Helper {
	public:
	Helper();
	~Helper();

	static double dotProduct(double a[], double b[], int length);
	static double RELUFunction(double input, double bias);
	static double calculateLoss(double value1, double value2);
	static std::vector<double> arrayToVector(double array[], int arraySize);

	static double randomDouble(double min, double max);
	static int randomInt(int min, int max);
};

