#pragma once

#include <vector>
#include "DecisionStump.h"
#include <math.h>

class AdaBoost
{
private:
	vector<float> alphas;
	vector<DecisionStump> weak_classifiers;
	float classification_error;

public:
	vector<float> Boost(Data dataset, DecisionStump classifier,int T);
	vector<float> WeightTrimmingBoost(Data dataset, DecisionStump classifier, int T, float beta);
	vector<float> WeightTrimmingBoost(Data dataset, DecisionStump classifier, int T, float beta, int max_cut_samples);
	float FindQuantile(vector<float> weights, float beta);
	void PrintError();
	void PrintResult();
	void Reset();

	AdaBoost();
	~AdaBoost();
};

