#include "pch.h"
#include "DecisionStump.h"

DecisionStump::DecisionStump()
{
}


DecisionStump::~DecisionStump()
{
}

int DecisionStump::Classify(Data dataset,int decisionAttribute, int sample, float decisionCondition, bool greaterThan)
{
	int result;
	if (greaterThan)
	{
		if (dataset.dataContainer[decisionAttribute][sample] > decisionCondition) result = 1;
		else result = -1;
	}
	else
	{
		if (dataset.dataContainer[decisionAttribute][sample] <= decisionCondition) result = 1;
		else result = -1;
	}
	return result;
}

int DecisionStump::Classify(Data dataset, int sample)
{
	int result;

	int decisionAttribute = trainedParameters.attr;
	float decisionCondition = trainedParameters.threshold;
	bool greaterThan = trainedParameters.isGreaterThan;

	if (greaterThan)
	{
		if (dataset.dataContainer[decisionAttribute][sample] > decisionCondition) result = 1;
		else result = -1;
	}
	else
	{
		if (dataset.dataContainer[decisionAttribute][sample] <= decisionCondition) result = 1;
		else result = -1;
	}
	return result;
}

void DecisionStump::Train(Data dataset)
{
	float error = FLT_MAX, err_temp;
	float threshold, prev_threshold = FLT_MAX;
	int idx, idx_next;
	int d;
	int m = dataset.dataContainer.size();
	int n = dataset.output.size();

	for (int i=0; i < m; i++)
	{
		for (int j=0; j<n; j++)
		{
			idx = dataset.sortedIndices[i][j];
			if(j + 1 < dataset.sortedIndices[i].size()) idx_next = dataset.sortedIndices[i][j + 1];
			else idx_next = dataset.sortedIndices[i][j];
			threshold = (dataset.dataContainer[i][idx] + dataset.dataContainer[i][idx_next]) / 2;

			err_temp = 0;
			if (threshold != prev_threshold)
			{
				for (int k = 0; k < n; k++)
				{
					d = Classify(dataset, i, k, threshold, false);
					if (d != dataset.output[k]) err_temp += 1.0f;
				}
				if (err_temp < error)
				{
					trainedParameters.attr = i;
					trainedParameters.threshold = threshold;
					trainedParameters.isGreaterThan = false;
					error = err_temp;
					break;
				}
			}

			err_temp = 0;		
			if (threshold != prev_threshold)
			{
				for (int k = 0; k < n; k++)
				{
					d = Classify(dataset, i, k, threshold, true);
					if (d != dataset.output[k]) err_temp += 1.0f;
				}
				if (err_temp < error)
				{
					trainedParameters.attr = i;
					trainedParameters.threshold = threshold;
					trainedParameters.isGreaterThan = true;
					error = err_temp;
					break;
				}
			}

			prev_threshold = threshold;
		}
	}
}

void DecisionStump::Train(Data dataset, vector<float> weights)
{
	float error = FLT_MAX, err_temp;
	float threshold, prev_threshold = FLT_MAX;;
	int idx, idx_next;
	int d;
	int m = dataset.dataContainer.size();
	int n = dataset.output.size();

	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
		{
			idx = dataset.sortedIndices[i][j];
			if (j + 1 < dataset.sortedIndices[i].size()) idx_next = dataset.sortedIndices[i][j + 1];
			else idx_next = dataset.sortedIndices[i][j];
			threshold = (dataset.dataContainer[i][idx] + dataset.dataContainer[i][idx_next]) / 2;

			err_temp = 0;
			if (threshold != prev_threshold)
			{
				for (int k = 0; k < n; k++)
				{
					d = Classify(dataset, i, k, threshold, false);
					if (d != dataset.output[k]) err_temp += weights[k];
				}
				if (err_temp < error)
				{
					trainedParameters.attr = i;
					trainedParameters.threshold = threshold;
					trainedParameters.isGreaterThan = false;
					error = err_temp;
				}
			}

			err_temp = 0;
			if (threshold != prev_threshold)
			{
				for (int k = 0; k < n; k++)
				{
					d = Classify(dataset, i, k, threshold, true);
					if (d != dataset.output[k]) err_temp += weights[k];
				}
				if (err_temp < error)
				{
					trainedParameters.attr = i;
					trainedParameters.threshold = threshold;
					trainedParameters.isGreaterThan = true;
					error = err_temp;
				}
			}

			prev_threshold = threshold;
		}
	}
}