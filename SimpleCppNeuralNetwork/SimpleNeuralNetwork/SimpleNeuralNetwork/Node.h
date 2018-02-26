#pragma once
#include <vector>
#include "Connection.h"
class Node {
	public:
	Node();
	~Node();

	std::vector<Connection*>& getInputs();
	std::vector<Connection*>& getOutputs();

	void setInputs(std::vector<Connection*> i);
	void setOutputs(std::vector<Connection*> o);

	void setValue(double v);
	double getValue();

	void setBias(double b);
	double getBias();

	void changeNeutralStatus();
	bool isNeutral();

	void addOutput(Connection* c);
	void addInput(Connection* c);




	private:
	std::vector<Connection*> inputs;
	std::vector<Connection*> outputs;

	double value;
	double bias;

	bool Neutral;


};

