#pragma once

#include <vector>
#include "DecisionStump.h"

class AdaBoost
{
private:
//	DecisionStump classifier;
//	vector<float> weights;

public:
	void Train(DecisionStump classifier);

	AdaBoost();
	~AdaBoost();
};

