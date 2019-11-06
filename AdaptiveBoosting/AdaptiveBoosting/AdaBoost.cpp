#include "pch.h"
#include "AdaBoost.h"


AdaBoost::AdaBoost()
{
}


AdaBoost::~AdaBoost()
{
}


vector<float> AdaBoost::Boost(Data dataset, DecisionStump classifier, int T)
{
	cout << "Training in progress..." << endl;

	vector<float> weights;
	int n = dataset.output.size();
	vector<float> y = dataset.output;
	vector<float> d;

	for (int i=0; i<n; i++)
	{
		weights.push_back(1.0/n);
	}
	
	for (int t = 0; t < T; t++)
	{
		classifier.Train(dataset, weights);

		for (int i = 0; i < n; i++)
		{
			d.push_back(classifier.Classify(dataset, i));
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
			result = result + alpha[t] * weakClassifiers[t].Classify(dataset, i);
		}
		if (result > 0) d.push_back(1);
		else d.push_back(-1);

		result = 0;
	}
	return d;
}

void AdaBoost::PrintResult()
{
	cout << "Training complete! Here are the results: " << endl << endl;
	cout << "Alpha values" << endl;
	for (int i = 0; i < alpha.size(); i++)
	{
		cout << alpha[i] << "	";
	}
	cout << endl << endl;
	cout << "Classifiers' trained parameters" << endl;
	for (int i = 0; i < weakClassifiers.size(); i++)
	{
		cout << "Attribute: " << weakClassifiers[i].trainedParameters.attr << "	" << "Threshold: " << weakClassifiers[i].trainedParameters.threshold << "	";
		cout << "Inequality direction: ";
		if (weakClassifiers[i].trainedParameters.isGreaterThan)
		{
			cout << ">" << endl;
		}
		else
		{
			cout << "<" << endl;
		}
	}
	cout << endl;
}