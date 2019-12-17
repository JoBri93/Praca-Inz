// AdaptiveBoosting.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include "pch.h"
#include "Data.h"
#include "DecisionStump.h"
#include "AdaBoost.h"
#include <time.h>
#include <iostream>

using namespace std;

Data training_set[3], testing_set[3];
DecisionStump classifier;
AdaBoost adaboost;

int main()
{
	srand(time(NULL));
	clock_t begin, end;
	double time_spent;
	int iterations[3] = { 400, 400, 20 };
	int max_samples[3] = { 8, 10, 10 };
	
	training_set[0].LoadFile("../Data/training_set1-cryotherapy.txt");
	testing_set[0].LoadFile("../Data/testing_set1-cryotherapy.txt");

	training_set[1].LoadFile("../Data/training_set2-fertility.txt");
	testing_set[1].LoadFile("../Data/testing_set2-fertility.txt");

	training_set[2].LoadFile("../Data/training_set3-parkinsons.txt");
	testing_set[2].LoadFile("../Data/testing_set3-parkinsons.txt");

    cout << "Welcome to ADABOOST!\n" << endl; 

	for (int i = 0; i < 3; i++)
	{
		cout << "Dataset " << i+1 << endl << endl;

		begin = clock();
		adaboost.Boost(training_set[i], testing_set[i], classifier, iterations[i]);
		end = clock();
		time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
		adaboost.PrintError();
		cout << "Time: " << time_spent << " seconds" << endl << endl;

		adaboost.Reset();

		begin = clock();
		adaboost.WeightTrimmingBoost(training_set[i], testing_set[i], classifier, iterations[i], 0.1f, max_samples[i]);
		end = clock();
		time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
		adaboost.PrintError();
		cout << "Time: " << time_spent << " seconds" << endl << endl;

		adaboost.Reset();

		begin = clock();
		adaboost.WeightTrimmingBoost(training_set[i], testing_set[i], classifier, iterations[i]);
		end = clock();
		time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
		adaboost.PrintError();
		cout << "Time: " << time_spent << " seconds" << endl << endl;

		adaboost.Reset();
	}
}