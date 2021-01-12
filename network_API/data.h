#pragma once

#include "global.h"

using namespace Global;
using namespace std;

class Data
{

private:
	double*	_input;
	double*	_output;
	
	int		_count;
	int		_inputSize;
	int		_outputSize;

	double*	_selectionScaling;
	double* _selectionCounts;

	int*	_currentRandomIndex;
	int		_currentIndex;
	int		_currentCount;

	double* _currentInput;
	double* _currentOutput;

	int		_errorBatchCount;
	double* _errorInput;
	double* _errorOutput;

public:

	Data(double* input, double* output, int count, int batchSize, int inputSize, int outputSize, int errorBatchCount);
	~Data();

	int		TotalCases();

	void	NewSelectionRound();
	bool	NextTrainingSet();
	
	double* Input();
	double* OutputExpected();

	double* ErrorInput();
	double* ErrorExpectedOutput();
};
