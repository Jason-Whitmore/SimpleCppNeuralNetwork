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

