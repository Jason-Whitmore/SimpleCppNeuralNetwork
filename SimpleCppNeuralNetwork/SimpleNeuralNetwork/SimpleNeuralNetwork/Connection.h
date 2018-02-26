#pragma once
class Connection {

	public:
		Connection();
		~Connection();


		void setValue(double v);
		double getValue();

		void setWeight(double w);
		double getWeight();

	private:

		double value;
		double weight;
		

};

