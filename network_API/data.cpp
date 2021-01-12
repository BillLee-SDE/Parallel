#include "data.h"
#include "cuda.cuh"
#include <random>
#include <time.h>

using namespace std;

static double * temp;

void CUDA_Read_Data(double* cudaValues, double** target, int size, int offset)
{
	if (*target != nullptr) { delete *target; }
	*target = new double[size];
	Cuda::CUDA_Array_CopyDeviceToHost(&cudaValues[offset], size * sizeof(double), *target);
}

void shuffle_array(int size, int* indexes)
{
	default_random_engine randomGenerator(std::random_device{}());
	mt19937 randomEngine((randomGenerator)());
	uniform_int_distribution<> randomDistribution(0, INT32_MAX);

	for (int i = size - 1; i > 0; i--)
	{
		int j = randomDistribution(randomEngine) % i;
		int temp = indexes[j];
		indexes[j] = indexes[i];
		indexes[i] = temp;
	}
}

Data::Data(double * input, double * output, int count, int batchSize, int inputSize, int outputSize, int errorBatchCount)
{
	_count = count;
	_inputSize = inputSize;
	_outputSize = outputSize;

	_errorBatchCount = errorBatchCount;

	_input = Cuda::CUDA_Array_Allocate<double>(_count * _inputSize);
	_output = Cuda::CUDA_Array_Allocate<double>(_count * _outputSize);

	_errorInput = Cuda::CUDA_Array_Allocate<double>(_errorBatchCount * batchSize * _inputSize);
	_errorOutput = Cuda::CUDA_Array_Allocate<double>(_errorBatchCount * batchSize * _outputSize);

	Cuda::CUDA_Array_CopyHostToDevice(input, _count * _inputSize * sizeof(double), _input);
	Cuda::CUDA_Array_CopyHostToDevice(output, _count * _outputSize * sizeof(double), _output);

	_selectionScaling = new double[_count];
	_selectionCounts = new double[_count];

	for (int i = 0; i < _count; i++) 
	{
		_selectionCounts[i] = 0.0;
		_selectionScaling[i] = (double)i / (double)_count + 1.0;
	}

	_currentRandomIndex = new int[_count * 2];
	NewSelectionRound();

	int* tempArray = new int[_count];
	for (int i = 0; i < _count; i++) { tempArray[i] = i; }
	shuffle_array(_count, tempArray);

	for (int i = 0; i < batchSize * _errorBatchCount; i++)
	{
		int index = tempArray[i];
		Cuda::CUDA_Array_CopyDeviceToDevice(&_input[index * _inputSize], _inputSize * sizeof(double), &_errorInput[i * _inputSize]);
		Cuda::CUDA_Array_CopyDeviceToDevice(&_output[index * _outputSize], _outputSize * sizeof(double), &_errorOutput[i * _outputSize]);
	}

	delete tempArray;
	//NextTrainingSet();
}

Data::~Data()
{
	Cuda::CUDA_Array_Free((void**)&_input);
	Cuda::CUDA_Array_Free((void**)&_output);
	Cuda::CUDA_Array_Free((void**)&_errorInput);
	Cuda::CUDA_Array_Free((void**)&_errorOutput);

	delete _currentRandomIndex;
	delete _selectionScaling;
	delete _selectionCounts;
}

int Data::TotalCases()
{
	double result = 0.0;
	for (int i = 0; i < _count; i++)
	{
		result += _selectionScaling[i];
	}

	return (int)(result + 0.5);
}

void Data::NewSelectionRound()
{
	_currentCount = 0;
	_currentIndex = 0;

	for (int i = 0; i < _count; i++)
	{
		_selectionCounts[i] += _selectionScaling[i];
		for (; _selectionCounts[i] > 0.5; _selectionCounts[i] -= 1.0)
		{
			_currentRandomIndex[_currentCount++] = i;
		}
	}

	shuffle_array(_currentCount, _currentRandomIndex);
}

bool Data::NextTrainingSet()
{
	if (_currentIndex >= _currentCount)
	{
		return false;
	}

	_currentInput = &_input[_currentRandomIndex[_currentIndex] * _inputSize];
	_currentOutput = &_output[_currentRandomIndex[_currentIndex] * _outputSize];

	_currentIndex++;

	return true;
}

//bool Data::NextTrainingSet()
//{
//	if (_selectionCurrentCount <= 0) { return false; }
//
//	while (true) 
//	{
//		int index = rand() % _count;
//		if (_selectionCountsRounded[index] > 0)
//		{
//			_selectionCountsRounded[index]--;
//			_selectionCurrentCount--;
//			_selectionCounts[index] = _selectionCounts[index] - 1.0;
//
//			_currentInput = &_input[index * _inputSize];
//			_currentOutput = &_output[index * _outputSize];
//
//			return true;
//		}
//	}
//
//}

double * Data::Input()
{
	return _currentInput;
}

double * Data::OutputExpected()
{
	return _currentOutput;
}

double * Data::ErrorInput()
{
	return _errorInput;
}

double * Data::ErrorExpectedOutput()
{
	return _errorOutput;
}
