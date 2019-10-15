// AdaptiveBoosting.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include "pch.h"
#include "DecisionStump.h"
#include "AdaBoost.h"

#include <iostream>

using namespace std;

DecisionStump classifier;

int main()
{
    cout << "Welcome to ADABOOST!\n"; 
	classifier.LoadFile("../Data/Test_data.txt");
	classifier.SelectOutput(3);
	classifier.Train();
	classifier.SaveFile("../ClassifiedData/Test_data_NEW.txt");
}