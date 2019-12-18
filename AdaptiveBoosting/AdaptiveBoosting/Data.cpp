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
		categories_container.push_back(new_line);
	}

	vector<float> temp1;
	while (getline(file, line))
	{
		istringstream iss(line);
		while (getline(iss, new_line, ';'))
		{
			temp1.push_back(strtof(new_line.c_str(), 0));
		}
		data_container.push_back(temp1);
		temp1.clear();
	}

	file.close();

	SplitSamples(0.7);

	TransposeDataMatrix(data_container);
	SelectOutput(data_container, output, data_container.size()-1);

	TransposeDataMatrix(training_set);
	CreateSortedIndexesMatrix();
	SelectOutput(training_set, training_output, training_set.size() - 1);

	TransposeDataMatrix(testing_set);
	SelectOutput(testing_set, testing_output, testing_set.size() - 1);
	
}

bool Data::SaveFile(string filename)
{
	TransposeDataMatrix(data_container);

	ofstream file(filename);
	if (file.fail())
		return 1;

	for (int i = 0; i < categories_container.size(); i++)
	{
		file << categories_container[i];
		file << ";";
	}
	file << endl;
	for (int j = 0; j < data_container.size(); j++)
	{
		for (int k = 0; k < data_container[j].size(); k++)
		{
			file << data_container[j][k];
			file << ";";
		}
		file << endl;
	}
	file.close();
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

void Data::SelectOutput(vector<vector<float>> &set, vector<float> &d, int attribute)
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
	set.erase(set.begin() + attribute);
}

void Data::SortIndexes(const vector<vector<float>> &X, vector<int> &idx, int feature)
{
	sort(idx.begin(), idx.end(), [&X, &feature](size_t i1, size_t i2) {return X[feature][i1] < X[feature][i2]; });
}

void Data::CreateSortedIndexesMatrix()
{
	vector<int> Indices(training_set[0].size());
	iota(begin(Indices), end(Indices), 0);

	for (int i = 0; i < training_set.size(); i++)
	{
		vector<int> sorted = Indices;
		SortIndexes(training_set, sorted, i);
		sorted_indices.push_back(sorted);
	}
}

void Data::SplitSamples(float percent)
{
	int n = data_container.size() * percent;
	int m = data_container.size() - n;
	int idx = rand() % data_container.size();
	vector<int> indices;
	int i = 0;

	while (i < n)
	{
		if (std::find(indices.begin(), indices.end(), idx) != indices.end())
		{
			idx = rand() % data_container.size();
		}
		else
		{
			indices.push_back(idx);
			training_set.push_back(data_container[idx]);
			i++;
			idx = rand() % data_container.size();
		}
	}

	for (int j = 0; j < data_container.size(); j++)
	{
		if (std::find(indices.begin(), indices.end(), j) == indices.end())
		{
			testing_set.push_back(data_container[j]);
		}
	}
}
