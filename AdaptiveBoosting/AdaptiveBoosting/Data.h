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
	
	vector<vector<int>> sorted_indices;
	vector<float> output;

	bool LoadFile(string filename);
	bool SaveFile(string filename);
	void CreateSortedIndexesMatrix();
	void SelectOutput(int attribute);

	Data();
	~Data();
};

