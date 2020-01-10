#pragma once

class Data
{
private:
	vector<string> categories;
	vector<vector<float>> data;

	vector<vector<float>> training_set;
	vector<vector<float>> testing_set;

	void SortIndexes(const vector<vector<float>> &X, vector<int> &idx, int feature);
	void TransposeDataMatrix(vector<vector<float>> &b);
	void SplitSamples(float percent);
	void SelectOutput(vector<vector<float>> set, vector<vector<float>> &x, vector<float> &d, int attribute);

public:
	vector<vector<float>> input;
	vector<float> output;

	vector<vector<float>> training_input;
	vector<float> training_output;	
	vector<vector<int>> training_set_sorted_indices;

	vector<vector<float>> testing_input;
	vector<float> testing_output;

	bool LoadFile(string filename);
	void CreateSortedIndexesMatrix();
	void ReloadTrainingAndTestingSets();

	Data();
	~Data();
};

