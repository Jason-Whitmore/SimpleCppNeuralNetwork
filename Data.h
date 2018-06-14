#pragma once
class Data {
	public:
	Data(int rows, int columns);
	~Data();

	double* getRow(int rowIndex);

	double getIndex(int row, int column);

	private:

	int numRows;
	int numCols;

	double* array;
};

