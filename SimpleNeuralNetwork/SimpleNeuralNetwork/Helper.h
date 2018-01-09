#pragma once
class Helper {
	public:
	Helper();
	~Helper();

	static int randomNumber(int a, int b);
	static double randomNumber(double a, double b);

	static double calculateLoss(double value, double target);
	static double activationFunctionRELU(double sum, double bias);
};

