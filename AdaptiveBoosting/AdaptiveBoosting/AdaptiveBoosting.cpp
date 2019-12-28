// AdaptiveBoosting.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include "pch.h"
#include "Data.h"
#include "DecisionStump.h"
#include "AdaBoost.h"
#include <time.h>
#include <iostream>

using namespace std;

Data dataset[3];
DecisionStump classifier;
AdaBoost adaboost;

int main()
{
	srand(time(NULL));
	clock_t begin, end;
	double time_spent;

	//iterations = { {200, 400, 800}, {200, 400, 800}, {20, 40, 60} };
	//max_samples = { {5, 10, 15}, {5, 10, 15}, {5, 10, 15}};
	
	int idx, iterations, max_samples;
	float beta;

	dataset[0].LoadFile("../Data/dataset1-cryotherapy.txt");
	dataset[1].LoadFile("../Data/dataset2-fertility_diagnosis.txt");
	dataset[2].LoadFile("../Data/dataset3-parkinsons.txt");

    cout << "Welcome to ADABOOST!\n" << endl;
	cout << "Available datasets:" << endl;
	cout << "1. Cryotherapy (" << dataset[0].categories_container.size() - 1 << " attributes, " << dataset[0].output.size() << " samples)" << endl;
	cout << "2. Fertility diagnosis (" << dataset[1].categories_container.size() - 1 << " attributes, " << dataset[1].output.size() << " samples)" << endl;
	cout << "3. Parkinsons (" << dataset[2].categories_container.size() - 1 << " attributes, " << dataset[2].output.size() << " samples)" << endl << endl;

	cout << "Choose dataset: ";
	cin >> idx;

	if (idx < 1 || idx > 3)
	{
		cout << "There is no dataset " << idx << "!" << endl;
	}
	else
	{
		idx = idx - 1;
		cout << "Enter number of iterations: ";
		cin >> iterations;

		cout << "For WeightTrimmingBoost: enter beta value (between 0.01 and 0.1): ";
		cin >> beta;
		if (beta < 0.01f || beta > 0.1f)
		{
			cout << "Wrong beta value!" << endl;
		}
		else
		{
			cout << "For WeightTrimmingBoost (type 1): enter max number of samples to cut: ";
			cin >> max_samples;

			cout << endl << "Dataset " << idx + 1 << ": " << iterations << " iterations, beta value - " << beta << ", samples to cut - " << max_samples << endl << endl;

			begin = clock();
			adaboost.Boost(dataset[idx], classifier, iterations);
			end = clock();
			time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
			adaboost.PrintError();
			cout << "Time: " << time_spent << " seconds" << endl << endl;

			adaboost.Reset();

			begin = clock();
			adaboost.WeightTrimmingBoost(dataset[idx], classifier, iterations, beta, max_samples);
			end = clock();
			time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
			adaboost.PrintError();
			cout << "Time: " << time_spent << " seconds" << endl << endl;

			adaboost.Reset();

			begin = clock();
			adaboost.WeightTrimmingBoost(dataset[idx], classifier, iterations, beta);
			end = clock();
			time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
			adaboost.PrintError();
			cout << "Time: " << time_spent << " seconds" << endl << endl;

			adaboost.Reset();
		}
	}
}