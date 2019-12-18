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
	int iterations[3] = { 800, 800, 40 };
	int max_samples[3] = { 10, 10, 10 };
	
	dataset[0].LoadFile("../Data/dataset1-cryotherapy.txt");
	dataset[1].LoadFile("../Data/dataset2-fertility_diagnosis.txt");
	dataset[2].LoadFile("../Data/dataset3-parkinsons.txt");

    cout << "Welcome to ADABOOST!\n" << endl; 
	
	for (int i = 0; i < 3; i++)
	{
		cout << "Dataset " << i+1 << endl << endl;

		begin = clock();
		adaboost.Boost(dataset[i], classifier, iterations[i]);
		end = clock();
		time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
		adaboost.PrintError();
		cout << "Time: " << time_spent << " seconds" << endl << endl;

		adaboost.Reset();

		begin = clock();
		adaboost.WeightTrimmingBoost(dataset[i], classifier, iterations[i], 0.1f, max_samples[i]);
		end = clock();
		time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
		adaboost.PrintError();
		cout << "Time: " << time_spent << " seconds" << endl << endl;

		adaboost.Reset();

		begin = clock();
		adaboost.WeightTrimmingBoost(dataset[i], classifier, iterations[i]);
		end = clock();
		time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
		adaboost.PrintError();
		cout << "Time: " << time_spent << " seconds" << endl << endl;

		adaboost.Reset();
	}
}