#pragma once

#include <vector>
#include "DecisionStump.h"
#include <math.h>

class AdaBoost
{
private:
//	DecisionStump classifier;
//	vector<float> weights;

public:
	void Start(DecisionStump classifier,int T);

	AdaBoost();
	~AdaBoost();
};

