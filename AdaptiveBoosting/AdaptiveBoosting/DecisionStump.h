#pragma once

#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <vector>

using namespace std;

class DecisionStump
{
private:
	void SortIndexes(const vector<vector<float>> &X, vector<int> &idx, int feature);

public:
	vector<string> categoriesContainer;
	vector<vector<float>> dataContainer;
	string decisionAttribute;
	float decisionCondition;
	vector<int> classificationResult;
	vector<vector<int>> sortedIndices;
	vector<float> output;

	bool LoadFile(string filename);
	bool SaveFile(string filename);
	
	void selectOutput(int attribute);
	void CreateSortedIndexesMatrix();
	int Classify(int decisionAttribute, int sample, float decisionCondition, bool greaterThan);
	void Train();



	DecisionStump();
	~DecisionStump();
};

