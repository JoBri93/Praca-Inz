// AdaptiveBoosting.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include "pch.h"
#include "Data.h"
#include "DecisionStump.h"
#include "AdaBoost.h"
#include "Test.h"
#include <time.h>

Test test;

int main()
{
	srand(time(NULL));

	cout << "Welcome to ADABOOST!\n" << endl;
	test.TakeParameters();
	test.PerformTest();
	test.SaveResults();
}