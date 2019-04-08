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
	vector<float> temp;
	while (getline(file, line))
	{
		istringstream iss(line);
		while (getline(iss, new_line, ';'))
		{
			temp.push_back(strtof(new_line.c_str(), 0));
		}
		dataContainer.push_back(temp);
		temp.clear();
	}
	file.close();

	cout << "CATEGORIES:" << endl;
	for (int i = 0; i < categoriesContainer.size(); i++)
	{
		cout << i << ". " << categoriesContainer[i] << endl;
	}
}

void DecisionStump::ClassifyData(int decisionAttribute, float decisionCondition, bool greaterThan)
{
	for (int i=0; i<dataContainer.size(); i++)
	{
		if (greaterThan)
		{
			if (dataContainer[i][decisionAttribute] > decisionCondition) classificationResult.push_back(1);
			else classificationResult.push_back(0);
		}
		else
		{
			if (dataContainer[i][decisionAttribute] < decisionCondition) classificationResult.push_back(1);
			else classificationResult.push_back(0);
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