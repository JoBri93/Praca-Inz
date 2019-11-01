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
	void Start(Data data1, DecisionStump classifier,int T);

	AdaBoost();
	~AdaBoost();
};

