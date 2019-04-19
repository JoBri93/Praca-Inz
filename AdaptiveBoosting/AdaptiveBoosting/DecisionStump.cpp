#include "pch.h"
#include "DecisionStump.h"

DecisionStump::DecisionStump()
{
}


DecisionStump::~DecisionStump()
{
}

bool DecisionStump::LoadFile(string filename)
{
	ifstream file;
	file.open(filename.c_str(), ios::in);
	if (file.fail())
		return 1;

	string line, new_line;
	getline(file, line);
	istringstream iss(line);
	while (getline(iss, new_line, ';'))
	{
		categoriesContainer.push_back(new_line);
	}

	vector<float> temp1;
	vector<vector<float>> temp2;
	while (getline(file, line))
	{
		istringstream iss(line);
		while (getline(iss, new_line, ';'))
		{
			temp1.push_back(strtof(new_line.c_str(), 0));
		}
		temp2.push_back(temp1);
		temp1.clear();
	}

	vector<float> temp3;
	for (int j = 0; j < temp2[0].size(); j++)
	{
		for (int i = 0; i < temp2.size(); i++)
		{
			temp3.push_back(temp2[i][j]);
		}
		dataContainer.push_back(temp3);
		temp3.clear();
	}

	file.close();

	cout << "CATEGORIES:" << endl;
	for (int i = 0; i < categoriesContainer.size(); i++)
	{
		cout << i << ". " << categoriesContainer[i] << endl;
	}
}

void DecisionStump::selectOutput(int attribute)
{
	for (int i = 0; i < dataContainer[attribute].size(); i++)
	{
		output.push_back(dataContainer[attribute][i]);
	}
}


void DecisionStump::SortIndexes(const vector<vector<float>> &X, vector<int> &idx, int feature)
{
	sort(idx.begin(), idx.end(), [&X, &feature](size_t i1, size_t i2) {return X[feature][i1] < X[feature][i2]; });
}

void DecisionStump::CreateSortedIndexesMatrix()
{
	vector<int> Indices(dataContainer[0].size());
	iota(begin(Indices), end(Indices), 0);

	for (int i = 0; i < dataContainer.size(); i++)
	{
		vector<int> sorted = Indices;
		SortIndexes(dataContainer, sorted, i);
		sortedIndices.push_back(sorted);
	}
}

int DecisionStump::Classify(int decisionAttribute, int sample, float decisionCondition, bool greaterThan)
{
	int result;
	if (greaterThan)
	{
		if (dataContainer[decisionAttribute][sample] > decisionCondition) result = 1;// classificationResult.push_back(1);
		else result = 0;// classificationResult.push_back(0);
	}
	else
	{
		if (dataContainer[decisionAttribute][sample] <= decisionCondition) result = 1;// classificationResult.push_back(1);
		else result = 0;// classificationResult.push_back(0);
	}
	return result;
}

void DecisionStump::Train()
{
	float error = FLT_MAX, err_temp;
	float threshold;
	int ind, ind_next;
	int d;
	for (int i=0; i<sortedIndices.size(); i++)
	{
		for (int j=0; j<sortedIndices[i].size(); j++)
		{
			ind = sortedIndices[i][j];
			ind_next = sortedIndices[i][j + 1];
			threshold = (dataContainer[i][ind] + dataContainer[i][ind_next]) / 2;
			d = Classify(i, ind, threshold, false);
			err_temp = d-output[ind];
		}
	}
}

bool DecisionStump::SaveFile(string filename)
{
	ofstream file(filename);
	if (file.fail())
		return 1;

	for (int i = 0; i < categoriesContainer.size(); i++)
	{
		file << categoriesContainer[i];
		file << ";";
	}
	file << "Decision Stump" << endl;
	for (int j = 0; j < dataContainer.size(); j++)
	{
		for (int k = 0; k < dataContainer[j].size(); k++)
		{
			file << dataContainer[j][k];
			file << ";";
		}
		file << classificationResult[j] << endl;
	}	
	file.close();
}