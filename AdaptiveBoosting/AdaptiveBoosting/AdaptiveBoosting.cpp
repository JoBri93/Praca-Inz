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

	int idx, iterations, max_samples, number_of_tests;
	float beta;

	dataset[0].LoadFile("../Data/dataset1-cryotherapy.txt");
	dataset[1].LoadFile("../Data/dataset2-fertility_diagnosis.txt");
	dataset[2].LoadFile("../Data/dataset3-parkinsons.txt");

    cout << "Welcome to ADABOOST!\n" << endl;
	cout << "Available datasets:" << endl;
	cout << "1. Cryotherapy (" << dataset[0].input.size() << " attributes, " << dataset[0].output.size() << " samples)" << endl;
	cout << "2. Fertility diagnosis (" << dataset[1].input.size() << " attributes, " << dataset[1].output.size() << " samples)" << endl;
	cout << "3. Parkinsons (" << dataset[2].input.size() << " attributes, " << dataset[2].output.size() << " samples)" << endl << endl;

	cout << "Choose dataset: ";
	cin >> idx;
	while (idx < 1 || idx > 3)
	{
		cout << "There is no dataset " << idx << "! Try again: ";
		cin >> idx;
	}
	idx = idx - 1;

	cout << "Enter number of iterations: ";
	cin >> iterations;
	while (iterations < 1)
	{
		cout << "Wrong number of iterations! Try again: ";
		cin >> iterations;
	}

	cout << "For WeightTrimmingBoost: enter beta value (between 0.01 and 0.1): ";
	cin >> beta;
	while (beta < 0.01f || beta > 0.1f)
	{
		cout << "Wrong beta value! Try again: ";
		cin >> beta;
	}

	cout << "For WeightTrimmingBoost (type 1): enter max number of samples to cut: ";
	cin >> max_samples;
	while (max_samples < 0)
	{
		cout << "Wrong number of samples! Try again: ";
		cin >> max_samples;
	}

	cout << "Enter number of tests to perform: ";
	cin >> number_of_tests;
	while (number_of_tests < 1)
	{
		cout << "Wrong number of tests! Try again: ";
		cin >> number_of_tests;
	}

	cout << endl << "Dataset " << idx + 1 << ": " << iterations << " iterations, beta value = " << beta << ", samples to cut = " << max_samples << endl << endl;
	for (int i = 0; i < number_of_tests; i++)
	{
		cout << "---------- Test " << i + 1 << " ----------" << endl << endl;

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
		dataset[idx].ReloadTrainingAndTestingSets();
	}
}