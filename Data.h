#pragma once
class Data {
	public:
	Data(int rows, int columns);
	Data();
	~Data();

	double* getRow(int rowIndex);

	double getIndex(int row, int column);

	void setIndex(int row, int column, double value);

	int getNumRows();
	int getNumCols();
	

	private:

	int numRows;
	int numCols;

	double** array;
};

