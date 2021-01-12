#pragma once

#define WIN32_LEAN_AND_MEAN             // Exclude rarely-used stuff from Windows headers
#include <windows.h>
#include "cuda_runtime.h"
#include "global.h"
#include "cuda.cuh"

#define NUM_RUNNING_AVERAGE_ERROR 30

namespace Calculation
{
	// Setup and adjust learning parameters

	void Initialize(double learningRate, int running_errors, Global::UpdateMethod update, int tempSize);

	void NextBatch();

	bool AdjustLearning(double newError);


	double Forward_Error(double * actual, double * expected, int size, Global::ErrorFunction ef);

	void Backward_ErrorSignal(double* actual, double* expected, double* target, int size, Global::ErrorFunction ef);


	void Calculation_Convolution_1D_Forward(
		int sizeBatch, int rows, int inputFilter, int inputColumns,
		double * input, int sizeWindow, double * weight, double * bias, 
		Global::ActivationFunction af, int outputFilter, int outputColumns,
		double * output);

	void Calculation_Convolution_1D_Backward(
		int sizeBatch, int sizeRow, int inputFilter, int inputColumns, 
		double* i_signal, double* input, int sizeWindow,
		double** weight, double** bias,	Global::ActivationFunction af, 
		int outputFilter, int outputColumns, double* activated, double* o_signal,
		CudaParameter parameterBias, CudaParameter parameterWeight);

	void Calculation_Pooling_1D_Forward(
		int sizeBatch, int sizeRow, int sizeFilter, int sizeUnpooled,
		double * input, int sizePooling, int sizePooled, Global::PoolingType pt,
		double* pooled, double* pooling);

	void Calculation_Pooling_1D_Backward(
		int sizeBatch, int sizeRow, int sizeFilter, int sizeUnpooled,
		double * unpooled_signal, int sizePooling, int sizePooled,
		double * pooled_signal, double* pooling);


	void Calculation_Convolution_2D_Forward(
		int sizeBatch, int inputFilter, int inputHeight, int inputWidth,
		double * input, int windowHeight, int windowWidth, double * weight, double * bias,
		Global::ActivationFunction af, int outputFilter, int outputHeight, int outputWidth,
		double * output);

	void Calculation_Convolution_2D_Backward(
		int sizeBatch, int inputFilter, int inputHeight, int inputWidth,
		double* i_signal, double* input, int windowHeight, int windowWidth,
		double** weight, double** bias, Global::ActivationFunction af,
		int outputFilter, int outputHeight, int outputWidth, double* activated, double* o_signal,
		CudaParameter parameterBias, CudaParameter parameterWeight);

	void Calculation_Pooling_2D_Forward(
		int sizeBatch, int sizeFilter, int unpooledHeight, int unpooledWidth,
		double * input, int poolingHeight, int poolingWidth, 
		int pooledHeight, int pooledWidth, Global::PoolingType pt, double* pooled, double* pooling);

	void Calculation_Pooling_2D_Backward(
		int sizeBatch, int sizeFilter, int unpooledHeight, int unpooledWidth,
		double * unpooled_signal, int poolingHeight, int poolingWidth, 
		int pooledHeight, int pooledWidth, double * pooled_signal, double* pooling);


	void Calculation_FullyConnected_Forward(
		int sizeBatch, int sizeInput, double * input,
		int * dropoutSelectInput, double scale, double ** weight, double ** bias,
		Global::ActivationFunction af, int sizeOutput, double* activated,
		CudaParameter parameter);

	void Calculation_FullyConnected_Backward(
		int sizeBatch, int sizeInput, double* i_signal, double * input,
		double ** weights, double ** bias, Global::ActivationFunction af,
		int sizeOutput, double* activated, int * dropoutSelectOutput,
		double* o_signal, CudaParameter parameterBias,
		CudaParameter parameterSignal, CudaParameter parameterWeight);


	void Convolution_Normalize_Convolution(
		int sizeBatch, int countFilter, int countRow, int sizeRow, double * data,
		double * mean, double * variance, double * sd, double * normalized,
		CudaParameter parameter);

};

