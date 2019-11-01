#include "pch.h"
#include "AdaBoost.h"


AdaBoost::AdaBoost()
{
}


AdaBoost::~AdaBoost()
{
}


void AdaBoost::Start(Data data1, DecisionStump classifier, int T)
{
	vector<float> weights;
	int n = data1.output.size();
	vector<float> y = data1.output;
	vector<int> d;
	for (int i=0; i<n; i++)
	{
		weights.push_back(1.0/n);
	}
	
	for (int t = 0; t < T; t++)
	{
		classifier.Train(data1, weights);

		for (int i = 0; i < n; i++)
		{
			d.push_back(classifier.Classify(data1, i));
		}

		float error = 0;

		for (int i = 0; i < n; i++)
		{
			if (d[i] != y[i]) error += weights[i];
		}

		float a;
		a = 0.5 * log((1 - error) / error);
		
		weakClassifiers.push_back(classifier);
		alpha.push_back(a);

		float w_sum = 0;
		for (int i = 0; i < n; i++)
		{
			weights[i] = weights[i] * exp(-a * y[i] * d[i]);
			w_sum += weights[i];
		}
		for (int i = 0; i < n; i++)
		{
			weights[i] = weights[i] / w_sum;
		}
		d.clear();
	}
	
	float result = 0;
	for (int i = 0; i < n; i++)
	{
		for (int t = 0; t < T; t++)
		{
			result = result + alpha[t] * weakClassifiers[t].Classify(data1, i);
		}
		d.push_back(result);
	}
}