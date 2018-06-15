#include "Data.h"


Data::Data(int rows, int columns) {
	numRows = rows;
	numCols = columns;

	array = new double*[rows];

	for (int i = 0; i < rows; i++) {
		array[i] = new double[columns];
	}


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
		r[i] = getIndex(rowIndex, i);

	}

	return r;
}


double Data::getIndex(int row, int column) {
	return array[row][column];
}


void Data::setIndex(int row, int column, double value) {
	array[row][column] = value;
}