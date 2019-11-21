// AdaptiveBoosting.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include "pch.h"
#include "Data.h"
#include "DecisionStump.h"
#include "AdaBoost.h"
#include <time.h>
#include <iostream>

using namespace std;

Data dataset;
DecisionStump classifier;
AdaBoost adaboost;

int main()
{
	srand(time(NULL));
	clock_t begin, end;
	double time_spent;
	int iterations = 40;

    cout << "Welcome to ADABOOST!\n" << endl; 
	dataset.LoadFile("../Data/test-fertility_diagnosis.txt");
	dataset.SelectOutput(2);

	begin = clock();
	adaboost.Boost(dataset, classifier, iterations);
	end = clock();
	time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
	adaboost.PrintResult();
	cout << "Time: " << time_spent << " seconds" << endl << endl;

	adaboost.Reset();
	
	begin = clock();
	adaboost.WeightTrimmingBoost(dataset, classifier, iterations);
	end = clock();
	time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
	adaboost.PrintResult();
	cout << "Time: " << time_spent << " seconds" << endl << endl;

	adaboost.Reset();

	begin = clock();
	adaboost.WeightTrimmingBoost(dataset, classifier, iterations, 0.1f, 10);
	end = clock();
	time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
	adaboost.PrintResult();
	cout << "Time: " << time_spent << " seconds" << endl << endl;
}