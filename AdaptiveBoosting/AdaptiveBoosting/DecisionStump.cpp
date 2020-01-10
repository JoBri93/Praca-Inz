#include "pch.h"
#include "DecisionStump.h"


DecisionStump::DecisionStump()
{
}


DecisionStump::~DecisionStump()
{
}

int DecisionStump::Classify(vector<vector<float>> set, int decision_attribute, int sample, float decision_condition, bool greater_than)
{
	int result;
	if (greater_than)
	{
		if (set[decision_attribute][sample] > decision_condition) result = 1;
		else result = -1;
	}
	else
	{
		if (set[decision_attribute][sample] <= decision_condition) result = 1;
		else result = -1;
	}
	return result;
}

int DecisionStump::Classify(vector<vector<float>> set, int sample)
{
	int result;

	int decision_attribute = trained_parameters.attr;
	float decision_condition = trained_parameters.threshold;
	bool greater_than = trained_parameters.is_greater_than;

	if (greater_than)
	{
		if (set[decision_attribute][sample] > decision_condition) result = 1;
		else result = -1;
	}
	else
	{
		if (set[decision_attribute][sample] <= decision_condition) result = 1;
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
	int m = dataset.training_input.size();
	int n = dataset.training_output.size();

	for (int i=0; i < m; i++)
	{
		for (int j=0; j < n-1; j++)
		{
			idx = dataset.training_set_sorted_indices[i][j];
			idx_next = dataset.training_set_sorted_indices[i][j + 1];
			threshold = (dataset.training_input[i][idx] + dataset.training_input[i][idx_next]) / 2;

			err_temp = 0;
			if (threshold != prev_threshold)
			{
				for (int k = 0; k < n; k++)
				{
					d = Classify(dataset.training_input, i, k, threshold, false);
					if (d != dataset.training_output[k]) err_temp += 1.0f;
				}
				if (err_temp < error)
				{
					trained_parameters.attr = i;
					trained_parameters.threshold = threshold;
					trained_parameters.is_greater_than = false;
					error = err_temp;
				}
			}

			err_temp = 0;		
			if (threshold != prev_threshold)
			{
				for (int k = 0; k < n; k++)
				{
					d = Classify(dataset.training_input, i, k, threshold, true);
					if (d != dataset.training_output[k]) err_temp += 1.0f;
				}
				if (err_temp < error)
				{
					trained_parameters.attr = i;
					trained_parameters.threshold = threshold;
					trained_parameters.is_greater_than = true;
					error = err_temp;
				}
			}

			prev_threshold = threshold;
		}
	}
}

void DecisionStump::Train(Data dataset, vector<float> weights)
{
	float error = FLT_MAX, err_temp;
	float threshold, prev_threshold = FLT_MAX;
	int idx, idx_next;
	int d;
	int m = dataset.training_input.size();
	int n = dataset.training_output.size();

	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n-1; j++)
		{
			idx = dataset.training_set_sorted_indices[i][j];
			idx_next = dataset.training_set_sorted_indices[i][j + 1];
			threshold = (dataset.training_input[i][idx] + dataset.training_input[i][idx_next]) / 2;

			err_temp = 0;
			if (threshold != prev_threshold)
			{
				for (int k = 0; k < n; k++)
				{
					d = Classify(dataset.training_input, i, k, threshold, false);
					if (d != dataset.training_output[k]) err_temp += weights[k];
				}
				if (err_temp < error)
				{
					trained_parameters.attr = i;
					trained_parameters.threshold = threshold;
					trained_parameters.is_greater_than = false;
					error = err_temp;
				}
			}

			err_temp = 0;
			if (threshold != prev_threshold)
			{
				for (int k = 0; k < n; k++)
				{
					d = Classify(dataset.training_input, i, k, threshold, true);
					if (d != dataset.training_output[k]) err_temp += weights[k];
				}
				if (err_temp < error)
				{
					trained_parameters.attr = i;
					trained_parameters.threshold = threshold;
					trained_parameters.is_greater_than = true;
					error = err_temp;
				}
			}

			prev_threshold = threshold;
		}
	}
}