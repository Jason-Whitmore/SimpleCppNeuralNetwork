#include "Connection.h"

class Node;

Connection::Connection() {
	weight = 0;
	value = 0;
}


Connection::~Connection() {

}


void Connection::setValue(double v) {
	value = v;
}

double Connection::getValue() {
	return value;
}

void Connection::setWeight(double w) {
	weight = w;
	
}

double Connection::getWeight() {
	return weight;
}
