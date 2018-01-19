#include "Node.h"



Node::Node() {
	value = 0;
	bias = 0;
	Neutral = true;
}


Node::~Node() {
}

std::vector<Connection>& Node::getInputs() {
	return inputs;
}

std::vector<Connection>& Node::getOutputs() {
	return outputs;
}

void Node::setInputs(std::vector<Connection> i) {
	inputs = i;
}

void Node::setOutputs(std::vector<Connection> o) {
	outputs = o;
}

void Node::setValue(double v) {
	value = v;
}

double Node::getValue() {
	return value;
}

void Node::setBias(double b) {
	bias = b;


}

double Node::getBias() {
	return bias;
}

void Node::changeNeutralStatus() {
	Neutral = !Neutral;
}

bool Node::isNeutral() {
	return Neutral;
}

void Node::addOutput(Connection c) {
	outputs.reserve(outputs.size() + 1);
	outputs.push_back(c);
	
}

void Node::addInput(Connection c) {
	inputs.reserve(inputs.size() + 1);
	inputs.push_back(c);
}
