#pragma once

#include "layer.h"

class Network
{
public:
	Network(char* iData, int iLength, int iRow, int iRowSize, int iOutputSizeVerify, int iMaxBatch, double iLearningRate, 
		Global::ErrorFunction iErrorFunction, Global::UpdateMethod iUpdateMethod, void(*iLogMessage)(char*));
	~Network();

	void AddTrainingSet(double* iInput, double* iOutputExpected);
	double Error(double * iInput, double * iOutputExpected);

	int BatchTrained();
	void BatchTrainedReset();

	void Calculate(double* iInput, int count, double* oOutput);

private:

	int _sizeInput;
	int _sizeOutput;
	int _maxBatch;

	int _trainingSet;
	int _trainingCount;

	Layer* _layerFirst;
	Layer* _layerLast;

	double* _expected;
	Global::ErrorFunction _errorFunction;
};
