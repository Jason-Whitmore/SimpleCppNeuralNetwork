#pragma once
#include <vector>
#include <iostream>
#include <fstream>
#include <string>

class Helper {

	
	public:
	Helper();
	~Helper();

	static int randomNumber(int a, int b);
	static double randomNumber(double a, double b);

	static double calculateLoss(double value, double target);
	static double activationFunctionRELU(double sum, double bias);


	static std::vector<double> parseLine(std::string target, std::string entrySeparator);
	
};

