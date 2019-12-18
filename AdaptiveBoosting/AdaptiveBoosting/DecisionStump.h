#pragma once

#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <vector>
#include "Data.h"

using namespace std;

struct parameters {
	int attr;
	float threshold;
	bool is_greater_than;
};

class DecisionStump
{
public:
	parameters trained_parameters;

	int Classify(vector<vector<float>> set, int decision_attribute, int sample, float decision_condition, bool greater_than);
	int Classify(vector<vector<float>> set, int sample);
	void Train(Data dataset);
	void Train(Data dataset, vector<float> weights);

	DecisionStump();
	~DecisionStump();
};

