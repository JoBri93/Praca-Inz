#include "pch.h"
#include "AdaBoost.h"


AdaBoost::AdaBoost()
{
}


AdaBoost::~AdaBoost()
{
}


void AdaBoost::Train(DecisionStump classifier)
{
	vector<float> weights;
	int n = classifier.output.size();
	for (int i=0; i<n; i++)
	{
		weights.push_back(1/n);
	}

	int attribute = classifier.trainedParameters.attr;

	float error = FLT_MAX, err_temp;
	float threshold;
	int ind, ind_next;
	int d;

	/* for (int j = 0; j < n; j++)
	{
		ind = classifier.sortedIndices[][j];
		if (j + 1 < n) ind_next = classifier.sortedIndices[i][j + 1];
		else ind_next = sortedIndices[i][j];
		threshold = (dataContainer[attribute][ind] + dataContainer[i][ind_next]) / 2;
		err_temp = 0;
		for (int k = 0; k < sortedIndices[i].size(); k++)
		{
			d = Classify(i, ind, threshold, false);
			err_temp += d - output[ind];
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

		for (int k = 0; k < sortedIndices[i].size(); k++)
		{
			d = Classify(i, ind, threshold, true);
			err_temp += d - output[ind];
		}

		if (err_temp < error)
		{
			trainedParameters.attr = ind;
			trainedParameters.threshold = threshold;
			trainedParameters.isGreaterThan = true;
			error = err_temp;
			break;
		}
	}*/

}
