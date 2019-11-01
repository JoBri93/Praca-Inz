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
	CreateSortedIndexesMatrix();
}

bool Data::SaveFile(string filename)
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
	for (int i = 0; i < dataContainer[attribute].size(); i++)
	{
		output.push_back(dataContainer[attribute][i]);
	}
	dataContainer.erase(dataContainer.begin() + attribute);
}

void Data::SortIndexes(const vector<vector<float>> &X, vector<int> &idx, int feature)
{
	sort(idx.begin(), idx.end(), [&X, &feature](size_t i1, size_t i2) {return X[feature][i1] < X[feature][i2]; });
}

void Data::CreateSortedIndexesMatrix()
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
