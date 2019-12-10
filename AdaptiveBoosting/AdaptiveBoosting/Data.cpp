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

	TransposeDataMatrix(data_container);
	CreateSortedIndexesMatrix();
	SelectOutput(data_container.size()-1);
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

void Data::SelectOutput(int attribute)
{
	for (int i = 0; i < data_container[attribute].size(); i++)
	{
		if (data_container[attribute][i] == 0)
		{
			output.push_back(-1);
		}
		else
		{
			output.push_back(data_container[attribute][i]);
		}
	}
	data_container.erase(data_container.begin() + attribute);
}

void Data::SortIndexes(const vector<vector<float>> &X, vector<int> &idx, int feature)
{
	sort(idx.begin(), idx.end(), [&X, &feature](size_t i1, size_t i2) {return X[feature][i1] < X[feature][i2]; });
}

void Data::CreateSortedIndexesMatrix()
{
	vector<int> Indices(data_container[0].size());
	iota(begin(Indices), end(Indices), 0);

	for (int i = 0; i < data_container.size(); i++)
	{
		vector<int> sorted = Indices;
		SortIndexes(data_container, sorted, i);
		sorted_indices.push_back(sorted);
	}
}
