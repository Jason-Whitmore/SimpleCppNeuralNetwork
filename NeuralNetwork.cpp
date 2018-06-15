#include "NeuralNetwork.h"




NeuralNetwork::NeuralNetwork(int inputs, int outputs, std::vector<int> layers) {


}

NeuralNetwork::~NeuralNetwork() {
	delete layers;
}

int main() {
	unsigned long long counter = 0;

	std::vector<int> test = std::vector<int>();

	while(true) {
		test.push_back(counter % 10);
		if (counter % 50000 == 0) {
			std::cout << test[counter] << std::endl;
		}
		
		counter++;
		
	}

}