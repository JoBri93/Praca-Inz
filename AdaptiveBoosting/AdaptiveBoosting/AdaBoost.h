#pragma once

#include <vector>
#include "DecisionStump.h"
#include <math.h>

class AdaBoost
{
private:
	vector<float> alphas;
	vector<DecisionStump> weak_classifiers;
	float err;

public:
	vector<float> Boost(Data dataset, DecisionStump classifier,int T);
	vector<float> WeightTrimmingBoost(Data dataset, DecisionStump classifier, int T);
	vector<float> WeightTrimmingBoost(Data dataset, DecisionStump classifier, int T, float beta, int sample_quantity);
	float FindQuantile(vector<float> weights);
	void PrintError();
	void PrintResult();
	void Reset();

	AdaBoost();
	~AdaBoost();
};

