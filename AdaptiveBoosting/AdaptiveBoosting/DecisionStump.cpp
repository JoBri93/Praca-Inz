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
	while (getline(file, line))
	{
		istringstream iss(line);
		while (getline(iss, new_line, ';'))
		{
			temp1.push_back(strtof(new_line.c_str(), 0));
		}
		dataContainer.push_back(temp1);
		temp1.clear();
	}

	file.close();

	TransposeDataMatrix(dataContainer);
}

void DecisionStump::TransposeDataMatrix(vector<vector<float>> &b)
{
	if (b.size() == 0)
		return;

	vector<vector<float> > trans_vec(b[0].size(), vector<float>());

	for (int i = 0; i < b.size(); i++)
	{
		for (int j = 0; j < b[i].size(); j++)
		{
			trans_vec[j].push_back(b[i][j]);
		}
	}

	b = trans_vec;
}

void DecisionStump::SelectOutput(int attribute)
{
	for (int i = 0; i < dataContainer[attribute].size(); i++)
	{
		output.push_back(dataContainer[attribute][i]);
	}
	dataContainer.erase(dataContainer.begin() + attribute);
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
		if (dataContainer[decisionAttribute][sample] > decisionCondition) result = 1;
		else result = -1;
	}
	else
	{
		if (dataContainer[decisionAttribute][sample] <= decisionCondition) result = 1;
		else result = -1;
	}
	return result;
}

int DecisionStump::Classify(int sample)
{
	int result;

	int decisionAttribute = trainedParameters.attr;
	float decisionCondition = trainedParameters.threshold;
	bool greaterThan = trainedParameters.isGreaterThan;

	if (greaterThan)
	{
		if (dataContainer[decisionAttribute][sample] > decisionCondition) result = 1;
		else result = -1;
	}
	else
	{
		if (dataContainer[decisionAttribute][sample] <= decisionCondition) result = 1;
		else result = -1;
	}
	return result;
}

void DecisionStump::Train()
{
	CreateSortedIndexesMatrix();

	float error = FLT_MAX, err_temp;
	float threshold;
	int ind, ind_next;
	int d;

	for (int i=0; i<sortedIndices.size(); i++)
	{
		for (int j=0; j<sortedIndices[i].size(); j++)
		{
			ind = sortedIndices[i][j];
			if(j + 1 < sortedIndices[i].size()) ind_next = sortedIndices[i][j + 1];
			else ind_next = sortedIndices[i][j];
			threshold = (dataContainer[i][ind] + dataContainer[i][ind_next]) / 2;
			err_temp = 0;
			for (int k=0; k<sortedIndices[i].size(); k++)
			{
				d = Classify(i, ind, threshold, false);
				err_temp += d - output[ind];
			}
			
			if (err_temp<error)
			{
				trainedParameters.attr = i;
				trainedParameters.threshold = threshold;
				trainedParameters.isGreaterThan = false;
				error = err_temp;
				break;
			}
			err_temp = 0;
			
			for (int k = 0; k < sortedIndices[i].size(); k++)
			{
				d = Classify(i, ind, threshold, true);
				err_temp += d - output[ind];
			}

			if (err_temp<error)
			{
				trainedParameters.attr = ind;
				trainedParameters.threshold = threshold;
				trainedParameters.isGreaterThan = true;
				error = err_temp;
				break;
			}
		}
	}
}

bool DecisionStump::SaveFile(string filename)
{
	TransposeDataMatrix(dataContainer);

	ofstream file(filename);
	if (file.fail())
		return 1;

	for (int i = 0; i < categoriesContainer.size(); i++)
	{
		file << categoriesContainer[i];
		file << ";";
	}
	file << endl;
	for (int j = 0; j < dataContainer.size(); j++)
	{
		for (int k = 0; k < dataContainer[j].size(); k++)
		{
			file << dataContainer[j][k];
			file << ";";
		}
		file << endl;
	}	
	file.close();
}