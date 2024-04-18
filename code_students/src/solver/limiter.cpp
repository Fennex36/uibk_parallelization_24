#include "solver/limiter.hpp"

#include <algorithm>
#include <iostream>

limiter_base::limiter_base() {}

limiter_minmod::limiter_minmod(double theta) { this->theta = theta; }

double limiter_minmod::compute(double first, double second, double third) {

	// TBD by students DONE
	double limit;
	if (first * second < 0 or first * third < 0) {
		limit = 0.;
	} else {
		if (first < 0) {
			limit = std::max(first, std::max(second, third));
		} else {
			limit = std::min(first, std::min(second, third));
		}
	}
	std::cout << "RESULT: " << first << " " << second << " " << third << " " << limit << std::endl;
	return limit;
}