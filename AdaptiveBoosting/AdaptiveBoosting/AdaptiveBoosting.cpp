// AdaptiveBoosting.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include "pch.h"
#include "Data.h"
#include "DecisionStump.h"
#include "AdaBoost.h"

#include <iostream>

using namespace std;

Data dataset;
DecisionStump classifier;
AdaBoost adaboost;

int main()
{
    cout << "Welcome to ADABOOST!\n"; 
	dataset.LoadFile("../Data/test-fertility_diagnosis.txt");
	dataset.SelectOutput(2);
	adaboost.Boost(dataset,classifier,10);
	adaboost.PrintResult();
}