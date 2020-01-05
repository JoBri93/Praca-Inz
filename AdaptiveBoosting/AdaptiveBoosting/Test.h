#pragma once

#include "Data.h"
#include "DecisionStump.h"
#include "AdaBoost.h"
#include <time.h>
#include <string>
#include <iostream>

using namespace std;

class Test
{
private:
	Data dataset[3];
	DecisionStump classifier;
	AdaBoost adaboost;

	int idx, iterations, max_samples, number_of_tests;
	float beta;

	vector<float> time_results[3];
	vector<float> classification_errors[3];

	bool SaveTimeResults(string filename);
	bool SaveErrorResults(string filename);

public:
	void TakeParameters();
	void PerformTest();
	void SaveResults();

	Test();
	~Test();
};

