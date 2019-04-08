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
	classifier.LoadFile("../Data/Absenteeism_at_work.txt");
	classifier.ClassifyData(8,30,false);
	classifier.SaveFile("../ClassifiedData/Absenteeism_at_work_NEW.txt");
}