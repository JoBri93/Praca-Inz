#pragma once

#include <vector>
#include "DecisionStump.h"
#include <math.h>

class AdaBoost
{
private:
	vector<float> alpha;
	vector<DecisionStump> weakClassifiers;

public:
	vector<float> Boost(Data dataset, DecisionStump classifier,int T);
	void PrintResult();

	AdaBoost();
	~AdaBoost();
};

