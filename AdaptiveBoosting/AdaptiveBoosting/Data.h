#pragma once

#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <vector>

using namespace std;

class Data
{
private:
	void SortIndexes(const vector<vector<float>> &X, vector<int> &idx, int feature);
	void TransposeDataMatrix(vector<vector<float>> &b);

public:
	vector<string> categories_container;
	vector<vector<float>> data_container;
	
	vector<vector<float>> training_set;
	vector<vector<float>> testing_set;
	
	vector<float> training_output;
	vector<float> testing_output;

	vector<vector<int>> sorted_indices;
	vector<float> output;

	bool LoadFile(string filename);
	bool SaveFile(string filename);
	void CreateSortedIndexesMatrix();
	void SelectOutput(vector<vector<float>> &set, vector<float> &d, int attribute);
	void SplitSamples(float percent);

	Data();
	~Data();
};

