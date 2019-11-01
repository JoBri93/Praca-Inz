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
	void CreateSortedIndexesMatrix();
	void TransposeDataMatrix(vector<vector<float>> &b);

public:
	vector<string> categoriesContainer;
	vector<vector<float>> dataContainer;
	
	vector<vector<int>> sortedIndices;
	vector<float> output;

	bool LoadFile(string filename);
	bool SaveFile(string filename);

	void SelectOutput(int attribute);

	Data();
	~Data();
};

