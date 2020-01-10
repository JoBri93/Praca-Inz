// AdaptiveBoosting.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include "pch.h"

Test test;

int main()
{
	srand(time(NULL));

	cout << "Welcome to ADABOOST!\n" << endl;
	test.PrintAvailableDatasets();
	test.TakeParameters();
	test.PerformTest();
	test.SaveResults();
}