#pragma once

#include "global.h"

#ifdef MACHINELEARNING_EXPORTS
#define MACHINELEARNING_API __declspec(dllimport)
#else
#define MACHINELEARNING_API __declspec(dllexport)
#endif

extern "C" {
	MACHINELEARNING_API void NeuronNetworkCalculation(
		int inputWidth, int inputHeight, int outputSize, bool batchNormalize, int batchSize, double startingLearningRate, Global::ErrorFunction ef,
		int configLength, char* configData, Global::UpdateMethod updateMethod, double* frequencyModifier, int errors_per_epoch, double max_error, bool classification,
		int trainingDataCount, double* trainingDataInput, double* trainingDataExpectedOutput,
		int validationCount, double* vadlidationInput, double* validationOutput,
		int forecastDataCount, double* forecastDataInput, double* forecastDataOutput,
		void(*iLogMessage)(char*));
}
