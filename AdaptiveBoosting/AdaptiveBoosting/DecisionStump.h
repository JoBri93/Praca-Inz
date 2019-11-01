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
	bool isGreaterThan;
};

class DecisionStump
{
public:
	parameters trainedParameters;

	int Classify(Data data1, int decisionAttribute, int sample, float decisionCondition, bool greaterThan);
	int Classify(Data data1, int sample);
	void Train(Data data1);
	void Train(Data data1, vector<float> weights);

	DecisionStump();
	~DecisionStump();
};

