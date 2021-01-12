// MachineLearning.cpp : Defines the exported functions for the DLL application.
//

#include "machineLearning.h"
#include <list>
#include <random>
#include "calculation.cuh"
#include "data.h"
#include "network.h"

#include <tchar.h>

double * temp = nullptr;

int classificationToIndex(int size, double* data)
{
	int result = 0;
	double max = INT32_MIN;
	for (int j = 0; j < size; j++)
	{
		if (data[j] > max)
		{
			max = data[j];
			result = j;
		}
	}

	return result;
}

int classificationNumber(double* result, int start, int size)
{
	int res = 0;
	for (int i = 1; i < size; i++)
	{
		if (result[start + i] > result[start + res])
		{
			res = i;
		}
	}
	return res;
}

void NeuronNetworkCalculation(
	int inputWidth, int inputHeight, int outputSize, bool batchNormalize, int batchSize, double startingLearningRate, Global::ErrorFunction ef, 
	int configLength, char* configData, Global::UpdateMethod updateMethod, double* frequencyModifier, int errors_per_epoch, double max_error, bool classification,
	int trainingDataCount, double* trainingDataInput, double* trainingDataExpectedOutput, 
	int validationCount, double* vadlidationInput, double* validationOutput,
	int forecastDataCount, double* forecastDataInput, double* forecastDataOutput, 
	void(*iLogMessage)(char*))
{
	Cuda::CUDA_Reset();
	char message[4096];

	int errorBatchCount = min(trainingDataCount / 10 / batchSize, 320 / batchSize) + 1;

	Data dataStream(trainingDataInput, trainingDataExpectedOutput, trainingDataCount, batchSize, inputHeight * inputWidth, outputSize, errorBatchCount);
	Network neuralNetwork(configData, configLength, inputWidth, inputHeight, outputSize, batchSize, startingLearningRate, ef, updateMethod, iLogMessage);

	int set_per_error = (dataStream.TotalCases() + errors_per_epoch - 1) / errors_per_epoch;
	bool keep_training = true;

	double epoch_error;

	for (int j = 0; j < 15 || (j < 20 && keep_training); j++) {
		int i = 0;
		epoch_error = 0;
		while (dataStream.NextTrainingSet())
		{
			if (i % set_per_error == 0)
			{
				double error = 0.0;
				for (int j = 0; j < errorBatchCount; j ++)
				{
					error += neuralNetwork.Error(&(dataStream.ErrorInput()[j * errorBatchCount * inputHeight * inputWidth]), &(dataStream.ErrorExpectedOutput()[j * errorBatchCount * outputSize]));
				}
				error = error / (double)errorBatchCount;

				sprintf_s(message, 256, "Epoch %d-%d, Error: %f", j, i, error);
				iLogMessage(message);
				keep_training &= Calculation::AdjustLearning(error);
				iLogMessage("\n");

				epoch_error += error;
			}

			neuralNetwork.AddTrainingSet(dataStream.Input(), dataStream.OutputExpected());
			i++;
		}
		dataStream.NewSelectionRound();
	}

	if (epoch_error / errors_per_epoch > max_error)
	{
		return;
	}

	if (forecastDataCount > 0)
	{
		neuralNetwork.Calculate(forecastDataInput, forecastDataCount, forecastDataOutput);
	}
}

