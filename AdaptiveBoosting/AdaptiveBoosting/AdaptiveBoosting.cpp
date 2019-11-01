// AdaptiveBoosting.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include "pch.h"
#include "Data.h"
#include "DecisionStump.h"
#include "AdaBoost.h"

#include <iostream>

using namespace std;

Data data1;
DecisionStump classifier;
AdaBoost adaboost;

int main()
{
    cout << "Welcome to ADABOOST!\n"; 
	data1.LoadFile("../Data/Test_data.txt");
	data1.SelectOutput(3);
	classifier.Train(data1);

	//classifier.LoadFile("../Data/Test_data.txt");
	//classifier.SelectOutput(3);
	//classifier.LoadFile("../Data/Test_data.txt");
	//classifier.SelectOutput(3);
	//classifier.Train();
	adaboost.Start(data1,classifier,3);
	//classifier.SaveFile("../ClassifiedData/Test_data_NEW.txt");
}