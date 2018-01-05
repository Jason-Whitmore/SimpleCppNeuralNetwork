#include "Node.h"



Node::Node() {
}


Node::~Node() {
}

std::vector<Connection> Node::getInputs() {
	return inputs;
}

std::vector<Connection> Node::getOutputs() {
	return outputs;
}

void Node::setInputs(std::vector<Connection> i) {
	inputs = i;
}

void Node::setOutPuts(std::vector<Connection> o) {
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

