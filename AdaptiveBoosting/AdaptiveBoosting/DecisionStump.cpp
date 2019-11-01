#include "pch.h"
#include "DecisionStump.h"

DecisionStump::DecisionStump()
{
}


DecisionStump::~DecisionStump()
{
}

int DecisionStump::Classify(Data data1,int decisionAttribute, int sample, float decisionCondition, bool greaterThan)
{
	int result;
	if (greaterThan)
	{
		if (data1.dataContainer[decisionAttribute][sample] > decisionCondition) result = 1;
		else result = -1;
	}
	else
	{
		if (data1.dataContainer[decisionAttribute][sample] <= decisionCondition) result = 1;
		else result = -1;
	}
	return result;
}

int DecisionStump::Classify(Data data1, int sample)
{
	int result;

	int decisionAttribute = trainedParameters.attr;
	float decisionCondition = trainedParameters.threshold;
	bool greaterThan = trainedParameters.isGreaterThan;

	if (greaterThan)
	{
		if (data1.dataContainer[decisionAttribute][sample] > decisionCondition) result = 1;
		else result = -1;
	}
	else
	{
		if (data1.dataContainer[decisionAttribute][sample] <= decisionCondition) result = 1;
		else result = -1;
	}
	return result;
}

void DecisionStump::Train(Data data1)
{
	float error = FLT_MAX, err_temp;
	float threshold;
	int ind, ind_next;
	int d;

	for (int i=0; i<data1.dataContainer.size(); i++)
	{
		for (int j=0; j<data1.sortedIndices[i].size(); j++)
		{
			ind = data1.sortedIndices[i][j];
			if(j + 1 < data1.sortedIndices[i].size()) ind_next = data1.sortedIndices[i][j + 1];
			else ind_next = data1.sortedIndices[i][j];
			threshold = (data1.dataContainer[i][ind] + data1.dataContainer[i][ind_next]) / 2;
			err_temp = 0;
			for (int k=0; k<data1.sortedIndices[i].size(); k++)
			{
				d = Classify(data1, i, ind, threshold, false);
				err_temp += d - data1.output[ind];
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
			
			for (int k = 0; k < data1.sortedIndices[i].size(); k++)
			{
				d = Classify(data1, i, ind, threshold, true);
				err_temp += d - data1.output[ind];
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

void DecisionStump::Train(Data data1, vector<float> weights)
{
	float error = FLT_MAX, err_temp;
	float threshold;
	int ind, ind_next;
	int d;

	for (int i = 0; i < data1.dataContainer.size(); i++)
	{
		for (int j = 0; j < data1.sortedIndices[i].size(); j++)
		{
			ind = data1.sortedIndices[i][j];
			if (j + 1 < data1.sortedIndices[i].size()) ind_next = data1.sortedIndices[i][j + 1];
			else ind_next = data1.sortedIndices[i][j];
			threshold = (data1.dataContainer[i][ind] + data1.dataContainer[i][ind_next]) / 2;
			err_temp = 0;
			for (int k = 0; k < data1.sortedIndices[i].size(); k++)
			{
				d = Classify(data1, i, ind, threshold, false);
				err_temp += weights[ind]*(d - data1.output[ind]);
			}

			if (err_temp < error)
			{
				trainedParameters.attr = i;
				trainedParameters.threshold = threshold;
				trainedParameters.isGreaterThan = false;
				error = err_temp;
				break;
			}
			err_temp = 0;

			for (int k = 0; k < data1.sortedIndices[i].size(); k++)
			{
				d = Classify(data1, i, ind, threshold, true);
				err_temp += weights[ind]*(d - data1.output[ind]);
			}

			if (err_temp < error)
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
