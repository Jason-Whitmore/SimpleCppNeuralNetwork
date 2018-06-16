#include "Data.h"


Data::Data(int rows, int columns) {
	numRows = rows;
	numCols = columns;

	array = new double*[rows];

	for (int i = 0; i < rows; i++) {
		array[i] = new double[columns];
	}


}

Data::Data() {
	numRows = -1;
	numCols = -1;

	array = nullptr;

}

Data::~Data() {
	delete array;
}

double* Data::getRow(int rowIndex) {
	if (rowIndex >= numRows) {
		return nullptr;
	}

	

	return array[rowIndex];
}


double Data::getIndex(int row, int column) {
	if (row >= numRows) {
		//exception here
	}

	if (column >= numCols) {
		//exception here
	}
	
	return array[row][column];
}


void Data::setIndex(int row, int column, double value) {
	if (row >= numRows) {
		//exception here
	}

	if (column >= numCols) {
		//exception here
	}

	array[row][column] = value;
}


int Data::getNumRows() {
	return numRows;
}


int Data::getNumCols() {
	return numRows;
}
