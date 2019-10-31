#pragma once

#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <vector>

using namespace std;

struct parameters {
	int attr;
	float threshold;
	bool isGreaterThan;
};

class DecisionStump
{
private:
	void SortIndexes(const vector<vector<float>> &X, vector<int> &idx, int feature);
	void CreateSortedIndexesMatrix();
	void TransposeDataMatrix(vector<vector<float> > &b);

public:
	parameters trainedParameters;

	vector<string> categoriesContainer;
	vector<vector<float>> dataContainer;
	string decisionAttribute;
	float decisionCondition;
	vector<int> classificationResult;
	vector<vector<int>> sortedIndices;
	vector<float> output;

	bool LoadFile(string filename);
	bool SaveFile(string filename);
	
	void SelectOutput(int attribute);
	int Classify(int decisionAttribute, int sample, float decisionCondition, bool greaterThan);
	int Classify(int sample);
	void Train();
	void Train(vector<float> weights);

	DecisionStump();
	~DecisionStump();
};

