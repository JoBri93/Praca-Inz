#include "pch.h"
#include "AdaBoost.h"

AdaBoost::AdaBoost()
{
}


AdaBoost::~AdaBoost()
{
}

float AdaBoost::FindQuantile(vector<float> weights, float beta)
{
	vector<float> w = weights;
	sort(w.begin(), w.end());
	auto last = unique(w.begin(), w.end());
	w.erase(last, w.end());

	float b;
	float sum = 0;
	float quantile = 0;
	
	for (int i = 0; i < w.size(); i++)
	{
		for (int j = 0; j < weights.size(); j++)
		{
			if (weights[j] < w[i])
			{
				sum += 1;
			}
		}
		b = sum / weights.size();
		if (b > beta)
		{
			break;
		}
		quantile = w[i];
		sum = 0;
	}

	return quantile;
}

vector<float> AdaBoost::Boost(Data dataset, DecisionStump classifier, int T)
{
	cout << "Boost: Training in progress..." << endl;

	vector<float> weights;
	int n = dataset.training_output.size();
	int m = dataset.testing_output.size();
	vector<float> y = dataset.training_output;
	vector<float> d;
	float a;
	float error;
	float w_sum;

	for (int i = 0; i < n; i++)
	{
		weights.push_back(1.0f/n);
	}
	
	for (int t = 0; t < T; t++)
	{
		classifier.Train(dataset, weights);

		for (int i = 0; i < n; i++)
		{
			d.push_back(classifier.Classify(dataset.training_input, i));
		}

		error = 0;

		for (int i = 0; i < n; i++)
		{
			if (d[i] != y[i]) error += weights[i];
		}

		a = 0.5 * log((1 - error) / error);
		
		weak_classifiers.push_back(classifier);
		alphas.push_back(a);

		w_sum = 0;
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
	y = dataset.testing_output;
	for (int i = 0; i < m; i++)
	{
		for (int t = 0; t < T; t++)
		{
			result = result + alphas[t] * weak_classifiers[t].Classify(dataset.testing_input, i);
		}
		if (result > 0) d.push_back(1);
		else d.push_back(-1);

		result = 0;
	}

	for (int i = 0; i < m; i++)
	{
		if (d[i] != y[i]) classification_error += 1.0f / m;
	}

	cout << "Training complete!" << endl;
	return d;
}

vector<float> AdaBoost::WeightTrimmingBoost(Data dataset, DecisionStump classifier, int T, float beta, int max_cut_samples)
{
	cout << "WeightTrimmingBoost (type 1): Training in progress..." << endl;

	vector<float> weights;
	int n = dataset.training_output.size();
	int m = dataset.testing_output.size();
	vector<float> y = dataset.training_output;
	vector<float> d;
	float a;
	float error;
	float w_sum;
	
	for (int i = 0; i < n; i++)
	{
		weights.push_back(1.0 / n);
	}
	
	Data reduced_dataset = dataset;
	vector<float> trimmed_weights = weights;
	int trim_idx;
	vector<int> trim_indices(n);

	for (int t = 0; t < T; t++)
	{	
		w_sum = 0;
		iota(begin(trim_indices), end(trim_indices), 0);
		sort(trim_indices.begin(), trim_indices.end(), [&weights](size_t i1, size_t i2) {return weights[i1] < weights[i2]; });

		for (int i = 0; i < n; i++)
		{
			w_sum += weights[trim_indices[i]];
			if (w_sum > beta || i > max_cut_samples)
			{
				trim_idx = i;
				break;		
			}		
		}

		trim_indices.erase(trim_indices.begin() + trim_idx, trim_indices.end());
		sort(trim_indices.begin(), trim_indices.end(), greater<int>());

		for (int i = 0; i < trim_indices.size() ; i++)
		{
			reduced_dataset.training_output.erase(reduced_dataset.training_output.begin() + trim_indices[i]);
			for (int j = 0; j < reduced_dataset.training_input.size(); j++)
			{
				reduced_dataset.training_input[j].erase(reduced_dataset.training_input[j].begin() + trim_indices[i]);
			}
			trimmed_weights.erase(trimmed_weights.begin() + trim_indices[i]);
		}

		reduced_dataset.training_set_sorted_indices.clear();
		reduced_dataset.CreateSortedIndexesMatrix();
		trim_indices.clear();
		trim_indices.resize(n);

		w_sum = 0;
		for (int i = 0; i < trimmed_weights.size(); i++)
		{
			w_sum += trimmed_weights[i];
		}
		for (int i = 0; i < trimmed_weights.size(); i++)
		{
			trimmed_weights[i] = trimmed_weights[i] / w_sum;
		}

		classifier.Train(reduced_dataset, trimmed_weights);

		for (int i = 0; i < n; i++)
		{
			d.push_back(classifier.Classify(dataset.training_input, i));
		}

		error = 0;
		for (int i = 0; i < n; i++)
		{
			if (d[i] != y[i]) error += weights[i];
		}

		a = 0.5 * log((1 - error) / error);

		weak_classifiers.push_back(classifier);
		alphas.push_back(a);

		w_sum = 0;
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

		reduced_dataset = dataset;
		trimmed_weights = weights;
	}

	float result = 0;
	y = dataset.testing_output;
	for (int i = 0; i < m; i++)
	{
		for (int t = 0; t < T; t++)
		{
			result = result + alphas[t] * weak_classifiers[t].Classify(dataset.testing_input, i);
		}
		if (result > 0) d.push_back(1);
		else d.push_back(-1);

		result = 0;
	}

	for (int i = 0; i < n; i++)
	{
		if (d[i] != y[i]) classification_error += 1.0f / n;
	}

	cout << "Training complete!" << endl;
	return d;
}

vector<float> AdaBoost::WeightTrimmingBoost(Data dataset, DecisionStump classifier, int T, float beta)
{
	cout << "WeightTrimmingBoost (type 2): Training in progress..." << endl;

	vector<float> weights;
	int n = dataset.training_output.size();
	int m = dataset.testing_output.size();
	vector<float> y = dataset.training_output;
	vector<float> d;
	float a;
	float error;
	float w_sum;

	for (int i = 0; i < n; i++)
	{
		weights.push_back(1.0 / n);
	}

	Data reduced_dataset = dataset;
	vector<float> trimmed_weights = weights;
	vector<float> trim_indices;

	for (int t = 0; t < T; t++)
	{
		for (int i = 0; i < n; i++)
		{
			if (weights[i] < FindQuantile(weights, beta))
			{
				trim_indices.push_back(i);
			}
		}

		sort(trim_indices.begin(), trim_indices.end(), greater<int>());

		if (!trim_indices.empty())
		{
			for (int i = 0; i < trim_indices.size(); i++)
			{
				reduced_dataset.training_output.erase(reduced_dataset.training_output.begin() + trim_indices[i]);
				for (int j = 0; j < reduced_dataset.training_input.size(); j++)
				{
					reduced_dataset.training_input[j].erase(reduced_dataset.training_input[j].begin() + trim_indices[i]);
				}
				trimmed_weights.erase(trimmed_weights.begin() + trim_indices[i]);
			}

			reduced_dataset.training_set_sorted_indices.clear();
			reduced_dataset.CreateSortedIndexesMatrix();
			trim_indices.clear();

			w_sum = 0;
			for (int i = 0; i < trimmed_weights.size(); i++)
			{
				w_sum += trimmed_weights[i];
			}
			for (int i = 0; i < trimmed_weights.size(); i++)
			{
				trimmed_weights[i] = trimmed_weights[i] / w_sum;
			}
		}

		classifier.Train(reduced_dataset, trimmed_weights);

		for (int i = 0; i < n; i++)
		{
			d.push_back(classifier.Classify(dataset.training_input, i));
		}

		error = 0;
		for (int i = 0; i < n; i++)
		{
			if (d[i] != y[i]) error += weights[i];
		}

		a = 0.5 * log((1 - error) / error);

		weak_classifiers.push_back(classifier);
		alphas.push_back(a);

		w_sum = 0;
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

		reduced_dataset = dataset;
		trimmed_weights = weights;
	}

	float result = 0;
	y = dataset.testing_output;
	for (int i = 0; i < m; i++)
	{
		for (int t = 0; t < T; t++)
		{
			result = result + alphas[t] * weak_classifiers[t].Classify(dataset.testing_input, i);
		}
		if (result > 0) d.push_back(1);
		else d.push_back(-1);

		result = 0;
	}

	for (int i = 0; i < m; i++)
	{
		if (d[i] != y[i]) classification_error += 1.0f / m;
	}

	cout << "Training complete!" << endl;
	return d;
}

void AdaBoost::PrintError()
{
	cout << "Error: " << classification_error << endl;
}

void AdaBoost::PrintResult()
{
	cout << "Here are the results: " << endl << endl;
	cout << "Alpha values" << endl;
	for (int i = 0; i < alphas.size(); i++)
	{
		cout << alphas[i] << "	";
	}
	cout << endl << endl;
	cout << "Classifiers' trained parameters" << endl;
	for (int i = 0; i < weak_classifiers.size(); i++)
	{
		cout << "Attribute: " << weak_classifiers[i].trained_parameters.attr << "	" << "Threshold: " << weak_classifiers[i].trained_parameters.threshold << "	";
		cout << "Inequality direction: ";
		if (weak_classifiers[i].trained_parameters.is_greater_than)
		{
			cout << ">" << endl;
		}
		else
		{
			cout << "<" << endl;
		}
	}
	cout << endl;
	PrintError();
}

void AdaBoost::Reset()
{
	alphas.clear();
	weak_classifiers.clear();
	classification_error = 0;
}