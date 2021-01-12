// TestML.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "machineLearning.h"

#include <fstream>
#include <string>
#include <assert.h>

#include <windows.h>
#include <iostream>

#define NUM_CODES			40
#define NUM_PREVIOUS_DAYS	120

using namespace std;

static char		_fileNameOutput[256];
static char*	_base;
static char		_code[16];
static char*	_network;
static double	_sd;

void printMessage(char* message)
{
	printf(message);
}

void build_fully_connected(char* parameters, int sizeOutput)
{
	int offset = 0;

	*(int*)&parameters[offset] = (int)Global::LayerType::CalculationFC;
	*(bool*)&parameters[offset + 4] = true;
	*(int*)&parameters[offset + 8] = (int)Global::ActivationFunction::ReLu;
	*(int*)&parameters[offset + 12] = 256;
	*(int*)&parameters[offset + 16] = 0;
	*(int*)&parameters[offset + 20] = 0;
	*(double*)&parameters[offset + 24] = -1.0;
	offset += 40;
	*(int*)&parameters[offset] = (int)Global::LayerType::CalculationFC;
	*(bool*)&parameters[offset + 4] = true;
	*(int*)&parameters[offset + 8] = (int)Global::ActivationFunction::Sigmoid;
	*(int*)&parameters[offset + 12] = sizeOutput;
	*(int*)&parameters[offset + 16] = 0;
	*(int*)&parameters[offset + 20] = 0;
	*(double*)&parameters[offset + 24] = -1.0;
}

void build_2D_convolution(char* parameters, int sizeOutput)
{
	int offset = 0;

	*(int*)&parameters[offset] = (int)Global::LayerType::CalculationC_2D;
	*(bool*)&parameters[offset + 4] = true;
	*(int*)&parameters[offset + 8] = 1;
	*(int*)&parameters[offset + 12] = 19;
	*(int*)&parameters[offset + 16] = (int)Global::ActivationFunction::ReLu;
	*(int*)&parameters[offset + 20] = 10;
	*(double*)&parameters[offset + 24] = 0.0;
	offset += 40;
	*(int*)&parameters[offset] = (int)Global::LayerType::Pooling_2D;
	*(bool*)&parameters[offset + 4] = false;
	*(int*)&parameters[offset + 8] = 1;
	*(int*)&parameters[offset + 12] = 3;
	*(int*)&parameters[offset + 16] = (int)Global::PoolingType::AbsoluteMax;
	*(int*)&parameters[offset + 20] = 0;
	*(double*)&parameters[offset + 24] = 0.0;
	offset += 40;
	*(int*)&parameters[offset] = (int)Global::LayerType::CalculationC_2D;
	*(bool*)&parameters[offset + 4] = true;
	*(int*)&parameters[offset + 8] = 1;
	*(int*)&parameters[offset + 12] = 20;
	*(int*)&parameters[offset + 16] = (int)Global::ActivationFunction::ReLu;
	*(int*)&parameters[offset + 20] = 10;
	*(double*)&parameters[offset + 24] = 0.0;
	offset += 40;
	*(int*)&parameters[offset] = (int)Global::LayerType::Pooling_2D;
	*(bool*)&parameters[offset + 4] = false;
	*(int*)&parameters[offset + 8] = 1;
	*(int*)&parameters[offset + 12] = 3;
	*(int*)&parameters[offset + 16] = (int)Global::PoolingType::AbsoluteMax;
	*(int*)&parameters[offset + 20] = 0;
	*(double*)&parameters[offset + 24] = 0.0;
	offset += 40;

	*(int*)&parameters[offset] = (int)Global::LayerType::CalculationFC;
	*(bool*)&parameters[offset + 4] = true;
	*(int*)&parameters[offset + 8] = (int)Global::ActivationFunction::ReLu;
	*(int*)&parameters[offset + 12] = 512;
	*(int*)&parameters[offset + 16] = 0;
	*(int*)&parameters[offset + 20] = 0;
	*(double*)&parameters[offset + 24] = -1.0;
	offset += 40;
	*(int*)&parameters[offset] = (int)Global::LayerType::CalculationFC;
	*(bool*)&parameters[offset + 4] = true;
	*(int*)&parameters[offset + 8] = (int)Global::ActivationFunction::Sigmoid;
	*(int*)&parameters[offset + 12] = sizeOutput;
	*(int*)&parameters[offset + 16] = 0;
	*(int*)&parameters[offset + 20] = 0;
	*(double*)&parameters[offset + 24] = -1.0;
}

void build_1D_convolution(char* parameters, int sizeOutput)
{
	int offset = 0;

	*(int*)&parameters[offset] = (int)Global::LayerType::CalculationC_1D;
	*(bool*)&parameters[offset + 4] = true;
	*(int*)&parameters[offset + 8] = 19;
	*(int*)&parameters[offset + 12] = (int)Global::ActivationFunction::ReLu;
	*(int*)&parameters[offset + 16] = 10;
	*(int*)&parameters[offset + 20] = 0;
	*(double*)&parameters[offset + 24] = 0.0;
	offset += 40;
	*(int*)&parameters[offset] = (int)Global::LayerType::Pooling_1D;
	*(bool*)&parameters[offset + 4] = false;
	*(int*)&parameters[offset + 8] = 3;
	*(int*)&parameters[offset + 12] = (int)Global::PoolingType::AbsoluteMax;
	*(int*)&parameters[offset + 16] = 0;
	*(int*)&parameters[offset + 20] = 0;
	*(double*)&parameters[offset + 24] = 0.0;
	offset += 40;
	*(int*)&parameters[offset] = (int)Global::LayerType::CalculationC_1D;
	*(bool*)&parameters[offset + 4] = true;
	*(int*)&parameters[offset + 8] = 20;
	*(int*)&parameters[offset + 12] = (int)Global::ActivationFunction::ReLu;
	*(int*)&parameters[offset + 16] = 10;
	*(int*)&parameters[offset + 20] = 0;
	*(double*)&parameters[offset + 24] = 0.0;
	offset += 40;
	*(int*)&parameters[offset] = (int)Global::LayerType::Pooling_1D;
	*(bool*)&parameters[offset + 4] = false;
	*(int*)&parameters[offset + 8] = 3;
	*(int*)&parameters[offset + 12] = (int)Global::PoolingType::AbsoluteMax;
	*(int*)&parameters[offset + 16] = 0;
	*(int*)&parameters[offset + 20] = 0;
	*(double*)&parameters[offset + 24] = 0.0;
	offset += 40;

	*(int*)&parameters[offset] = (int)Global::LayerType::CalculationFC;
	*(bool*)&parameters[offset + 4] = true;
	*(int*)&parameters[offset + 8] = (int)Global::ActivationFunction::ReLu;
	*(int*)&parameters[offset + 12] = 512;
	*(int*)&parameters[offset + 16] = 0;
	*(int*)&parameters[offset + 20] = 0;
	*(double*)&parameters[offset + 24] = -1.0;
	offset += 40;
	*(int*)&parameters[offset] = (int)Global::LayerType::CalculationFC;
	*(bool*)&parameters[offset + 4] = true;
	*(int*)&parameters[offset + 8] = (int)Global::ActivationFunction::Sigmoid;
	*(int*)&parameters[offset + 12] = sizeOutput;
	*(int*)&parameters[offset + 16] = 0;
	*(int*)&parameters[offset + 20] = 0;
	*(double*)&parameters[offset + 24] = -1.0;
}

int main(int argc, char *argv[])
{
	if (argc != 5) { return 0; }

	int numTraining = atoi(argv[1]);
	int numTesting = atoi(argv[2]);
	int networkType = atoi(argv[3]);
	int sizeOutput = atoi(argv[4]);
	
	char parameters[40 * 6];
	for (int i = 0; i < 40 * 6; i++) {
		parameters[i] = 0;
	}

	HANDLE handleTrainingInput = CreateFileMappingW(INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE, 0, sizeof(double) * numTraining * 4800, L"fs_training_input");
	HANDLE handleTrainingOutput = CreateFileMappingW(INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE, 0, sizeof(double) * numTraining * sizeOutput, L"fs_training_output");
	HANDLE handleTestingInput = CreateFileMappingW(INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE, 0, sizeof(double) * numTesting * 4800, L"fs_testing_input");
	HANDLE handleTestingOutput = CreateFileMappingW(INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE, 0, sizeof(double) * numTesting * sizeOutput, L"fs_testing_output");

	double* trainingInput = (double*)MapViewOfFile(handleTrainingInput, FILE_MAP_READ | FILE_MAP_WRITE, 0, 0, sizeof(double) * numTraining * 4800);
	double* trainingOutput = (double*)MapViewOfFile(handleTrainingOutput, FILE_MAP_READ | FILE_MAP_WRITE, 0, 0, sizeof(double) * numTraining * sizeOutput);
	double* testingInput = (double*)MapViewOfFile(handleTestingInput, FILE_MAP_READ | FILE_MAP_WRITE, 0, 0, sizeof(double) * numTesting * 4800);
	double* testingOutput = (double*)MapViewOfFile(handleTestingOutput, FILE_MAP_READ | FILE_MAP_WRITE, 0, 0, sizeof(double) * numTesting * sizeOutput);

	if (networkType == 0)
	{
		build_2D_convolution(parameters, sizeOutput);
	}
	else if (networkType == 1)
	{
		build_1D_convolution(parameters, sizeOutput);
	}

	if (networkType == 0 || networkType == 1) {
		NeuronNetworkCalculation(
			120, 40, sizeOutput, false, 64, 0.003, Global::ErrorFunction::CrossEntropy,
			6, parameters, Global::UpdateMethod::Adam, nullptr, 10, networkType == 0 ? 0.05 : 0.2, true,
			numTraining, trainingInput, trainingOutput,
			0, nullptr, nullptr,
			numTesting, testingInput, testingOutput, printMessage);
	}

	CloseHandle(handleTrainingInput);
	CloseHandle(handleTrainingOutput);
	CloseHandle(handleTestingInput);
	CloseHandle(handleTestingOutput);

	return 0;
}