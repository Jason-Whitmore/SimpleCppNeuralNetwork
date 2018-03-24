#pragma once
#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>

class Helper {

	
	public:
	Helper();
	~Helper();

	static int randomNumber(int a, int b);
	static double randomNumber(double a, double b);

	static double calculateLoss(double a, double b);
	static double activationFunctionRELU(double sum, double bias);


	
	static std::vector<double> parseLineDouble(std::string target, std::string entrySeparator);
	static std::vector<std::string> parseLineString(std::string target, std::string entrySeparator);

	
	static std::vector<std::vector<double>> csvToTable(std::string filePath, std::string rowSeparator, std::string entrySeparator, int rowStart, int rowEnd, int columnStart, int columnEnd);


	private:

	static std::vector<std::vector<std::string>> csvToVector(std::string filePath, std::string rowSeparator, std::string entrySeparator);
	static double calculateSimilarity(double a, double b);
};

