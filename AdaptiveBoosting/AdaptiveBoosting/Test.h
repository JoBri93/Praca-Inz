#pragma once

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
	void PrintAvailableDatasets();
	void TakeParameters();
	void PerformTest();
	void SaveResults();

	Test();
	~Test();
};

