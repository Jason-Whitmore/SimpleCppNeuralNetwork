#pragma once
#include <vector>
#include <random>
#include <map>
#include <cmath>
#include <string>
#include <iostream>
#include <thread>

class NNHelper {
	public:
	NNHelper();
	~NNHelper();

	static double dotProduct(double a[], double b[], int length);
	
	static double RELUFunction(double input, double bias);
	static double calculateLoss(double value1, double value2);
	static std::vector<double> arrayToVector(double array[], int arraySize);

	static double randomDouble(double min, double max);
	static int randomInt(int min, int max);
	static std::vector<std::string> split(std::string s, std::string splitter);
	static std::vector<double> stringToDoubleVector(std::vector<std::string> v);
	static std::vector<int> generateConfig(int numNodes, int minNodeCount);

	private:
	
	
	static bool contains(std::vector<int> v, std::vector<std::vector<int>> setOfVectors);
	
	static bool contains(std::string s, std::string targetString);
	
	static void multiThreadedDotProduct(int threads, double* result, double a[], double b[], int length);

	static void multiThreadedHelper(int threadNumber, int threadCount, double* result, double a[], double b[], int length);
};

