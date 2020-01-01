#include "pch.h"
#include "Data.h"


Data::Data()
{
}


Data::~Data()
{
}


bool Data::LoadFile(string filename)
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
		categories.push_back(new_line);
	}

	vector<float> temp1;
	while (getline(file, line))
	{
		istringstream iss(line);
		while (getline(iss, new_line, ';'))
		{
			temp1.push_back(strtof(new_line.c_str(), 0));
		}
		data.push_back(temp1);
		temp1.clear();
	}

	file.close();

	SplitSamples(0.7);

	TransposeDataMatrix(data);
	SelectOutput(data, input, output, data.size()-1);

	TransposeDataMatrix(training_input);
	CreateSortedIndexesMatrix();
	SelectOutput(training_input, training_input, training_output, training_input.size() - 1);

	TransposeDataMatrix(testing_set);
	SelectOutput(testing_set, testing_input, testing_output, testing_set.size() - 1);
}

void Data::TransposeDataMatrix(vector<vector<float>> &b)
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

void Data::SelectOutput(vector<vector<float>> set, vector<vector<float>> &x, vector<float> &d, int attribute)
{
	for (int i = 0; i < set[attribute].size(); i++)
	{
		if (set[attribute][i] == 0)
		{
			d.push_back(-1);
		}
		else
		{
			d.push_back(set[attribute][i]);
		}
	}
	x = set;
	x.erase(x.begin() + attribute);
}

void Data::SortIndexes(const vector<vector<float>> &X, vector<int> &idx, int feature)
{
	sort(idx.begin(), idx.end(), [&X, &feature](size_t i1, size_t i2) {return X[feature][i1] < X[feature][i2]; });
}

void Data::CreateSortedIndexesMatrix()
{
	vector<int> Indices(training_input[0].size());
	iota(begin(Indices), end(Indices), 0);

	for (int i = 0; i < training_input.size(); i++)
	{
		vector<int> sorted = Indices;
		SortIndexes(training_input, sorted, i);
		training_set_sorted_indices.push_back(sorted);
	}
}

void Data::SplitSamples(float percent)
{
	int n = data.size() * percent;
	int idx = rand() % data.size();
	vector<int> indices;
	
	int i = 0;
	while (i < n)
	{
		if (find(indices.begin(), indices.end(), idx) != indices.end())
		{
			idx = rand() % data.size();
		}
		else
		{
			indices.push_back(idx);
			training_input.push_back(data[idx]);
			i++;
			idx = rand() % data.size();
		}
	}

	for (int j = 0; j < data.size(); j++)
	{
		if (find(indices.begin(), indices.end(), j) == indices.end())
		{
			testing_set.push_back(data[j]);
		}
	}
}

void Data::ReloadTrainingAndTestingSets()
{
	training_input.clear();
	training_input.clear();
	training_output.clear();
	training_set_sorted_indices.clear();

	testing_set.clear();
	testing_input.clear();
	testing_output.clear();

	TransposeDataMatrix(data);
	SplitSamples(0.7);
	TransposeDataMatrix(data);

	TransposeDataMatrix(training_input);
	CreateSortedIndexesMatrix();
	SelectOutput(training_input, training_input, training_output, training_input.size() - 1);

	TransposeDataMatrix(testing_set);
	SelectOutput(testing_set, testing_input, testing_output, testing_set.size() - 1);
}