#include "pch.h"
#include "AdaBoost.h"


AdaBoost::AdaBoost()
{
}


AdaBoost::~AdaBoost()
{
}


void AdaBoost::Start(DecisionStump classifier, int T)
{
	vector<float> weights;
	int n = classifier.output.size();
	vector<float> y = classifier.output;
	vector<int> d;
	for (int i=0; i<n; i++)
	{
		weights.push_back(1.0/n);
	}
	
	for (int t = 0; t < T; t++)
	{
		classifier.Train(weights);

		for (int i = 0; i < n; i++)
		{
			d.push_back(classifier.Classify(i));
		}

		float error = 0;

		for (int i = 0; i < n; i++)
		{
			if (d[i] != y[i]) error += weights[i];
		}

		float alfa;
		alfa = 0.5 * log((1 - error) / error);

		float w_sum = 0;
		for (int i = 0; i < n; i++)
		{
			weights[i] = weights[i] * exp(-alfa * y[i] * d[i]);
			w_sum += weights[i];
		}
		for (int i = 0; i < n; i++)
		{
			weights[i] = weights[i] / w_sum;
		}
	}
}