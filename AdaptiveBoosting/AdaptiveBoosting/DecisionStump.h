#pragma once

#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>

using namespace std;

class DecisionStump
{
public:
	vector<string> categoriesContainer;
	vector<vector<float>> dataContainer;
	string decisionAttribute;
	float decisionCondition;
	vector<int> classificationResult;

	bool LoadFile(string filename);
	bool SaveFile(string filename);
	void ClassifyData(int decisionAttribute, float decisionCondition, bool greaterThan);

	DecisionStump();
	~DecisionStump();
};

