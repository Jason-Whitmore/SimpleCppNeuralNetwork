#include "Data.h"


Data::Data(int rows, int columns) {
	numRows = rows;
	numCols = columns;

	array = new double[rows * columns];

}

Data::~Data() {
	delete array;
}

double* Data::getRow(int rowIndex) {
	if (rowIndex > numRows - 1) {
		return nullptr;
	}

	double* r = new double[numCols];

	for(int i = 0; i < numCols; i++) {
		r[i] = array[(rowIndex * numCols) + i];

	}

	return r;
}

double Data::getIndex(int row, int column) {
	return array[(row * numCols) + column];
}
