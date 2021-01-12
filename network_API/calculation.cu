#include "calculation.cuh"
#include "device_launch_parameters.h"

#include <malloc.h>
#include <math.h>
#include <chrono> 

#define CUDA_CALCULATION_ADAM_BETA1			0.9
#define CUDA_CALCULATION_ADAM_BETA2			0.999
#define CUDA_CALCULATION_NEAR_ZERO			0.00000001
#define CUDA_CALCULATION_NORMALIZING_RATIO  0.95

namespace Calculation
{
	double*	_processingData1;
	double*	_processingData2;

	double	_learning_rate;
	double	_normalizing_running_ratio;

	int		_count_running_error;
	double*	_errors;
	double	_oldError;
	bool	_running_error_ready;
	int		_running_error_index;
	int		_flat_sacle_count;

	Global::UpdateMethod _update_method;
}

void Calculation::Initialize(double learningRate, int running_errors, Global::UpdateMethod update, int tempSize)
{
	_processingData1 = Cuda::CUDA_Array_Allocate<double>(tempSize);
	_processingData2 = Cuda::CUDA_Array_Allocate<double>(tempSize);

	_learning_rate = learningRate;
	_normalizing_running_ratio = 1.0;

	if (_errors)
	{
		delete _errors;
	}
	_count_running_error = NUM_RUNNING_AVERAGE_ERROR;
	_errors = new double[_count_running_error];
	_oldError = 0.0;
	_running_error_ready = false;
	_running_error_index = 0;

	_flat_sacle_count = 0;

	_update_method = update;
}

void Calculation::NextBatch()
{
	_normalizing_running_ratio *= CUDA_CALCULATION_NORMALIZING_RATIO;
}

double cuCalculation_regression(
	int size, int start, double * data, double * average)
{
	double x = 0.0;
	double y = 0.0;
	double x2 = 0.0;
	double xy = 0.0;
	for (int i = 0; i < NUM_RUNNING_AVERAGE_ERROR; i++)
	{
		int index = (start + i) % NUM_RUNNING_AVERAGE_ERROR;
		x += (double)(i + 1);
		y += data[index];
		x2 += (double)((i + 1) * (i + 1));
		xy += data[index] * (double)(i + 1);
	}

	*average = y / (double)size;
	return (xy * (double)NUM_RUNNING_AVERAGE_ERROR - y * x) / (x2 * size - x * x);
}

bool Calculation::AdjustLearning(double newError)
{
	if (_oldError == 0.0) { _oldError = newError; }
	if (_running_error_index == NUM_RUNNING_AVERAGE_ERROR - 1)
	{
		_running_error_ready = true;
	}
	_errors[_running_error_index++] = newError;

	_running_error_index = _running_error_index % NUM_RUNNING_AVERAGE_ERROR;

	if (_running_error_ready)
	{
		double average = 0.0;
		double result = cuCalculation_regression(
			_count_running_error, _running_error_index, _errors, &average);
		printf(", trend: %f", result);

		if (result > 0.0)
		{
			_flat_sacle_count++;
			if (_flat_sacle_count >= 2) 
			{ 
				printf(", early return");
				return false;
			}

			double scale = 2.5;
			printf(", scaled down learning rate", scale);
			_learning_rate = _learning_rate / scale;
			_running_error_index = 0;
			_running_error_ready = false;
		}
	}

	return true;
}


// Implementation of reduction

__device__ void cuda_Reduce_Dummy(int count)
{
	if (count <= 1) { return; }

	if (count > 128) { __syncthreads(); }

	if (count > 64) { __syncthreads(); }

	if (count > 32) { __syncthreads(); }

	if (count > 16) { __syncthreads(); }

	if (count > 8) { __syncthreads(); }

	if (count > 4) { __syncthreads(); }

	if (count > 2) { __syncthreads(); }

	if (count > 1) { __syncthreads(); }

	__syncthreads();
}

__device__ bool cuda_Reduce_Setup(
	int index, int sizeResult, int threadPerResult,
	int* indexResult, int* indexThread)
{
	if (index >= sizeResult * threadPerResult)
	{
		return false;
	}

	*indexResult = index / threadPerResult;
	*indexThread = index % threadPerResult;

	return true;
}

__device__ void cuda_Reduce(
	double* buffer, double input, int threadID, int count, double* result)
{
	int id = threadID % count;

	if (count <= 1)
	{
		*result = input;
		return;
	}

	buffer[id] = input;

	if (count > 128)
	{
		__syncthreads();
		if (id >= 128 && id < 256) { buffer[id - 128] += buffer[id]; }
	}

	if (count > 64)
	{
		__syncthreads();
		if (id >= 64 && id < 128) { buffer[id - 64] += buffer[id]; }
	}

	if (count > 32)
	{
		__syncthreads();
		if (id >= 32 && id < 64) { buffer[id - 32] += buffer[id]; }
	}

	if (count > 16)
	{
		__syncthreads();
		if (id >= 16 && id < 32) { buffer[id - 16] += buffer[id]; }
	}

	if (count > 8)
	{
		__syncthreads();
		if (id >= 8 && id < 16) { buffer[id - 8] += buffer[id]; }
	}

	if (count > 4)
	{
		__syncthreads();
		if (id >= 4 && id < 8) { buffer[id - 4] += buffer[id]; }
	}

	if (count > 2)
	{
		__syncthreads();
		if (id >= 2 && id < 4) { buffer[id - 2] += buffer[id]; }
	}

	if (count > 1)
	{
		__syncthreads();
		if (id >= 1 && id < 2) { buffer[id - 1] += buffer[id]; }
	}

	__syncthreads();
	if (id == 0) { *result = buffer[0]; }
}


// Activation and derivative functions

__device__ double cuda_activation(double x, Global::ActivationFunction af)
{
	switch (af)
	{
	case Global::Sigmoid:
		return 1.0 / (1.0 + exp(-x));
	case Global::ReLu:
		return x <= 0.0 ? 0.0 : x;
	case Global::LeakyReLu:
		return x <= 0.0 ? 0.01 * x : x;
	case Global::TanH:
		return 2.0 / (exp(-2.0 * x) + 1.0) - 1.0;
	default:
		return 0;
	}
}

__device__ double cuda_derivative(double y, Global::ActivationFunction af)
{
	switch (af)
	{
	case Global::Sigmoid:
		return y * (1.0 - y);
	case Global::ReLu:
		return y <= 0 ? 0 : 1;
	case Global::LeakyReLu:
		return y <= 0 ? 0.01 : 1;
	case Global::TanH:
		return 1.0 - pow(y, 2.0);
	default:
		return 0;
	}
}


// Parameter update functions

__global__ void cuda_Adam_Update(
	int size, double * grad, double * target, double * beta1power,
	double * beta2power, double * m1_running, double * m2_running,
	double learningRate, int * select, int selectSize)
{
	int index = blockDim.x * blockIdx.x + threadIdx.x;

	if (index < size)
	{
		if (select == nullptr || select[index % selectSize])
		{
			double gra = grad[index];

			beta1power[index] *= CUDA_CALCULATION_ADAM_BETA1;
			beta2power[index] *= CUDA_CALCULATION_ADAM_BETA2;

			double m1_r =
				m1_running[index] * CUDA_CALCULATION_ADAM_BETA1 +
				(1.0 - CUDA_CALCULATION_ADAM_BETA1) * gra;
			double m2_r =
				m2_running[index] * CUDA_CALCULATION_ADAM_BETA2 +
				(1.0 - CUDA_CALCULATION_ADAM_BETA2) * gra * gra;

			m1_running[index] = m1_r;
			m2_running[index] = m2_r;

			double m = m1_r / (1.0 - beta1power[index]);
			double v = m2_r / (1.0 - beta2power[index]);

			target[index] =
				target[index] +
				learningRate * m / (sqrt(v + CUDA_CALCULATION_NEAR_ZERO));
		}
	}
}

__global__ void cuda_SGD_Update(
	int size, double * grad, double * target, double * momentum,
	double learningRate, int * select, int selectSize)
{
	int index = blockDim.x * blockIdx.x + threadIdx.x;

	if (index < size)
	{
		if (select == nullptr || select[index % selectSize])
		{
			double gra = grad[index];

			double change = gra * learningRate + momentum[index] * 0.1;
			momentum[index] = change;

			target[index] = target[index] + change;
		}
	}
}

void update_by_grad(
	int size, double * grad, double ** parameters, int * select, int selectSize,
	double learning_rate, Global::UpdateMethod update_method)
{
	switch (update_method)
	{
	case Global::Adam:
		cuda_Adam_Update
			<< <Cuda::CUDA_Get_Block_Size(size), CUDA_CALCULATION_BLOCK_THREAD_SIZE >> >
			(size, grad, parameters[0], parameters[1], parameters[2],
				parameters[3], parameters[4], learning_rate, select, selectSize);
		break;
	case Global::SGD:
	default:
		cuda_SGD_Update
			<< <Cuda::CUDA_Get_Block_Size(size), CUDA_CALCULATION_BLOCK_THREAD_SIZE >> >
			(size, grad, parameters[0], parameters[3], learning_rate, select, selectSize);
		break;
	}
	Cuda::CUDA_Calculation_Error();
}


// Error and Signal functions

__global__ void cuda_Forward_Error(
	double* expected, double* actual, int size, Global::ErrorFunction ef,
	double* errors, int threadPerResult)
{
	extern __shared__ double buffer[];
	int index = blockDim.x * blockIdx.x + threadIdx.x;

	int dummy;
	if (!cuda_Reduce_Setup(index, 1, threadPerResult, &dummy, &dummy))
	{
		cuda_Reduce_Dummy(threadPerResult);
		return;
	}

	double result = 0.0;

	for (int i = 0; i < size; i += threadPerResult)
	{
		if (expected[i] == actual[i])
		{
			continue;
		}

		double act = actual[i];

		switch (ef)
		{
		case Global::CrossEntropy:
			if (act >= 1.0)
			{
				act = 0.999999;
			}
			else if (act <= 0.0)
			{
				act = 0.000001;
			}
			result += -expected[i] * log(act) - (1 - expected[i]) * log(1 - act);
			break;
		case Global::MeanSquare:
		default:
			double diff = expected[i] - act;
			result += diff * diff / 2;
			break;
		}
	}

	cuda_Reduce(
		&buffer[threadIdx.x - threadIdx.x % threadPerResult], result,
		threadIdx.x, threadPerResult, errors);
}

double Calculation::Forward_Error(double * actual, double * expected, int size, Global::ErrorFunction ef)
{
	double result = 0.0;

	CudaParameter parameter;
	parameter.Set(1, CUDA_CALCULATION_BLOCK_THREAD_SIZE);

	cuda_Forward_Error
		<< <parameter.CountBlock, CUDA_CALCULATION_BLOCK_THREAD_SIZE, CUDA_CALCULATION_BLOCK_THREAD_SIZE * sizeof(double) >> >
		(expected, actual, size, ef, _processingData1, parameter.ThreadPerResult);
	Cuda::CUDA_Calculation_Error();

	Cuda::CUDA_Array_CopyDeviceToHost(_processingData1, sizeof(double), &result);

	return result / (double)size;
}

__global__ void cuda_Backward_Signal(double* expected, double* actual, int size, Global::ErrorFunction ef, double* signal)
{
	int index = blockDim.x * blockIdx.x + threadIdx.x;

	if (index < size)
	{
		if (expected[index] == actual[index])
		{
			signal[index] = 0.0;
			return;
		}

		double act = actual[index];

		switch (ef)
		{
		case Global::CrossEntropy:
			if (act >= 1.0)
			{
				act = 0.999999;
			}
			else if (act <= 0.0)
			{
				act = 0.000001;
			}
			signal[index] = expected[index] / act - (1 - expected[index]) / (1 - act);
			break;
		case Global::MeanSquare:
		default:
			signal[index] = expected[index] - act;
			break;
		}
	}
}

void Calculation::Backward_ErrorSignal(double * actual, double * expected, double* target, int size, Global::ErrorFunction ef)
{
	cuda_Backward_Signal
		<< <Cuda::CUDA_Get_Block_Size(size), CUDA_CALCULATION_BLOCK_THREAD_SIZE >> >
		(expected, actual, size, ef, target);
	Cuda::CUDA_Calculation_Error();
}


// Convolution 1D calculation layer

__global__ void cuda_Calculation_Convolution_1D_Forward(
	int sizeBatch, int countRow, int inputFilter, int inputSizeRow,
	double* input, int sizeWindow, double* weight, double* bias,
	Global::ActivationFunction af, double* activated,
	int outputFilter, int outputSizeRow)
{
	int index = blockDim.x * blockIdx.x + threadIdx.x;

	int indexBatch = index;

	int indexColumn = indexBatch % outputSizeRow;
	indexBatch = indexBatch / outputSizeRow;

	int indexRow = indexBatch % countRow;
	indexBatch = indexBatch / countRow;

	int indexFilter = indexBatch % outputFilter;
	indexBatch = indexBatch / outputFilter;

	if (indexBatch < sizeBatch)
	{				
		double result = 0.0;

		for (int j = 0; j < inputFilter; j++)
		{
			for (int k = 0; k < sizeWindow; k++)
			{
				result +=
					input[
						indexBatch * inputFilter * countRow * inputSizeRow +
						j * countRow * inputSizeRow +
						indexRow * inputSizeRow + 
						indexColumn + k] *
					weight[
						indexFilter * inputFilter * countRow * sizeWindow +
						j * countRow * sizeWindow + 
						indexRow * sizeWindow +
						k];
			}
		}

		if (bias == nullptr)
		{
			activated[index] = cuda_activation(result, af);
		}
		else
		{
			activated[index] = cuda_activation(result + bias[indexFilter * countRow + indexRow], af);
		}
	}
}

void Calculation::Calculation_Convolution_1D_Forward(
	int sizeBatch, int rows, int inputFilter, int inputColumns,
	double * input, int sizeWindow, double * weight, double * bias, 
	Global::ActivationFunction af, int outputFilter, int outputColumns,
	double * output)
{
	cuda_Calculation_Convolution_1D_Forward
		<< <Cuda::CUDA_Get_Block_Size(sizeBatch * rows * outputFilter * outputColumns), CUDA_CALCULATION_BLOCK_THREAD_SIZE >> >
		(sizeBatch, rows, inputFilter, inputColumns, input,
			sizeWindow, weight, bias, af, output,
			outputFilter, outputColumns);
	Cuda::CUDA_Calculation_Error();
}
	
__global__ void cuda_Calculation_Convolution_1D_Backward_Derivative_Bias(
	int sizeBatch, int sizeRow, int sizeFilter, int sizeColumn,
	double * bias_grad,	Global::ActivationFunction af, 
	double * activiated, double * o_signal,	int threadPerResult)
{
	extern __shared__ double buffer[];
	int index = blockDim.x * blockIdx.x + threadIdx.x;

	int indexGroup = 0;
	int indexCalculation = 0;
	
	if (!cuda_Reduce_Setup(index, sizeFilter * sizeRow, threadPerResult,
		&indexGroup, &indexCalculation))
	{
		if (bias_grad != nullptr)
		{
			cuda_Reduce_Dummy(threadPerResult);
		}
		return;
	}

	int indexFilter = indexGroup / sizeRow;
	int indexRow = indexGroup % sizeRow;
	double result = 0.0;

	int sizeCalculation = sizeBatch * sizeColumn;
	
	for (int i = indexCalculation; i < sizeCalculation; i += threadPerResult)
	{
		int indexBatch = indexCalculation / sizeColumn;
		int indexColumn = indexCalculation % sizeColumn;

		int indexResult = 
			indexBatch * sizeFilter * sizeRow * sizeColumn + 
			indexFilter * sizeRow * sizeColumn + 
			indexRow * sizeColumn + 
			indexColumn;

		double tempSignal = o_signal[indexResult] * cuda_derivative(activiated[indexResult], af);

		if (bias_grad != nullptr)
		{
			result += tempSignal;
		}

		o_signal[indexResult] = tempSignal;
	}

	if (bias_grad != nullptr)
	{
		cuda_Reduce(&buffer[threadIdx.x - threadIdx.x % threadPerResult], result, threadIdx.x, threadPerResult, &bias_grad[indexGroup]);
	}
}
	
__global__ void cuda_Calculation_Convolution_1D_Backward_Signal(
	int sizeBatch, int sizeRow, int sizeInputFilter, int sizeInputColumn,
	int sizeWindow, int sizeOutputFilter, int sizeOutputColumn,
	double* i_signal, double* weight, double* o_signal)
{
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	int indexBatch = index;

	int indexInputColumn = indexBatch % sizeInputColumn;
	indexBatch /= sizeInputColumn;

	int indexRow = indexBatch % sizeRow;
	indexBatch /= sizeRow;

	int indexInputFilter = indexBatch % sizeInputFilter;
	indexBatch /= sizeInputFilter;

	if (indexBatch < sizeBatch)
	{
		double result = 0.0;

		for (int indexOutputFilter = 0; indexOutputFilter < sizeOutputFilter; indexOutputFilter++)
		{
			for (int i = 0; i < sizeWindow; i++)
			{
				int indexOutputColumn = indexInputColumn - i;
				if (indexOutputColumn >= 0 && indexOutputColumn < sizeOutputColumn)
				{
					result +=
						o_signal[
							indexBatch * sizeOutputFilter * sizeRow * sizeOutputColumn +
							indexOutputFilter * sizeRow * sizeOutputColumn +
							indexRow * sizeOutputColumn +
							indexOutputColumn] *
						weight[
							indexOutputFilter * sizeInputFilter * sizeRow * sizeWindow +
							indexInputFilter * sizeRow * sizeWindow +
							indexRow * sizeWindow + 
							i];
				}
			}
		}

		i_signal[index] = result;
	}
}
	
__global__ void cuda_Calculation_Convolution_1D_Backward_Weight_Grad(
	int sizeBatch, int sizeRow, int sizeInputFilter, int sizeInputColumn,
	int sizeWindow, int sizeOutputFilter, int sizeOutputColumn,
	double * input, double* weight_grad, double* o_signal, int threadPerResult)
{
	extern __shared__ double buffer[];
	int index = blockDim.x * blockIdx.x + threadIdx.x;

	int indexWeight = 0;
	int indexCalculation = 0;
	if (!cuda_Reduce_Setup(index, sizeOutputFilter * sizeInputFilter * sizeRow * sizeWindow, threadPerResult, &indexWeight, &indexCalculation)) 
	{ 
		cuda_Reduce_Dummy(threadPerResult);
		return; 
	}

	int indexOutputFilter = indexWeight;

	int indexInputFilter = indexOutputFilter % sizeInputFilter;
	indexOutputFilter = indexOutputFilter / sizeInputFilter;

	int indexRow = indexOutputFilter % sizeRow;
	indexOutputFilter = indexOutputFilter / sizeRow;

	int indexWindow = indexOutputFilter % sizeWindow;
	indexOutputFilter = indexOutputFilter / sizeWindow;

	double result = 0.0;

	int sizeCalc = sizeBatch * sizeOutputColumn;
	for (int indexCalc = indexCalculation; indexCalc < sizeCalc; indexCalc += threadPerResult)
	{
		int indexOutputColumn = indexCalc % sizeOutputColumn;
		int indexBatch = indexCalc / sizeOutputColumn;

		result +=
			o_signal[
				indexBatch * sizeOutputFilter * sizeRow * sizeOutputColumn +
				indexOutputFilter * sizeRow * sizeOutputColumn +
				indexRow * sizeOutputColumn +
				indexOutputColumn] *
			input[
				indexBatch * sizeInputFilter * sizeRow * sizeInputColumn +
				indexInputFilter * sizeRow * sizeInputColumn +
				indexRow * sizeInputColumn +
				indexOutputColumn + indexWindow];
	}

	cuda_Reduce(&buffer[threadIdx.x - threadIdx.x % threadPerResult], result, threadIdx.x, threadPerResult, &weight_grad[indexWeight]);
}

void Calculation::Calculation_Convolution_1D_Backward(
	int sizeBatch, int sizeRow, int inputFilter, int inputColumns, 
	double* i_signal, double* input, int sizeWindow,
	double** weight, double** bias,	Global::ActivationFunction af, 
	int outputFilter, int outputColumns, double* activated, double* o_signal,
	CudaParameter parameterBias, CudaParameter parameterWeight)
{
	cuda_Calculation_Convolution_1D_Backward_Derivative_Bias
		<< <parameterBias.CountBlock, CUDA_CALCULATION_BLOCK_THREAD_SIZE, CUDA_CALCULATION_BLOCK_THREAD_SIZE * sizeof(double) >> >
		(sizeBatch, sizeRow, outputFilter, outputColumns,
			bias[0] == nullptr ? nullptr : _processingData1, af,
			activated, o_signal, parameterBias.ThreadPerResult);
	Cuda::CUDA_Calculation_Error();

 	if (bias[0] != nullptr)
	{
		update_by_grad(
			sizeRow * outputFilter, _processingData1, bias,
			nullptr, 0, _learning_rate, _update_method);
	}

	if (i_signal)
	{
		cuda_Calculation_Convolution_1D_Backward_Signal
			<< <Cuda::CUDA_Get_Block_Size(sizeBatch * sizeRow * inputFilter * inputColumns), CUDA_CALCULATION_BLOCK_THREAD_SIZE >> >
			(sizeBatch, sizeRow, inputFilter, inputColumns, sizeWindow,
				outputFilter, outputColumns, i_signal, weight[0], o_signal);
		Cuda::CUDA_Calculation_Error();
	}

	cuda_Calculation_Convolution_1D_Backward_Weight_Grad
		<< <parameterWeight.CountBlock, CUDA_CALCULATION_BLOCK_THREAD_SIZE, CUDA_CALCULATION_BLOCK_THREAD_SIZE * sizeof(double) >> >
		(sizeBatch, sizeRow, inputFilter, inputColumns, sizeWindow,
			outputFilter, outputColumns, input, _processingData1, o_signal,
			parameterWeight.ThreadPerResult);
	Cuda::CUDA_Calculation_Error();

	update_by_grad(
		outputFilter * inputFilter * sizeRow * sizeWindow, _processingData1,
		weight, nullptr, 0, _learning_rate, _update_method);
}


// Convolutional 1D pooling layer

__global__ void cuda_Calculation_Pooling_1D_Forward(
	int sizeBatch, int sizeRow, int sizeFilter, int sizeUnpooled,
	double* input, int sizePooling, int sizePooled, Global::PoolingType pt,
	double* pooled, double* pooling)
{
	int index = blockDim.x * blockIdx.x + threadIdx.x;

	int indexBatch = index;

	int indexColumn = indexBatch % sizePooled;
	indexBatch = indexBatch / sizePooled;

	int indexRow = indexBatch % sizeRow;
	indexBatch = indexBatch / sizeRow;

	int indexFilter = indexBatch % sizeFilter;
	indexBatch = indexBatch / sizeFilter;

	if (indexBatch < sizeBatch)
	{
		int count = min(sizePooling, sizeUnpooled - indexColumn * sizePooling);

		if (pt == Global::PoolingType::AbsoluteMax)
		{
			int indexInput = -1;
			double result = 0;

			for (int i = 0; i < count; i++)
			{
				int newIndex = indexBatch * sizeFilter * sizeRow * sizeUnpooled +
					indexFilter * sizeRow * sizeUnpooled +
					indexRow * sizeUnpooled +
					indexColumn * sizePooling + i;
				pooling[newIndex] = 0.0;

				if (abs(input[newIndex]) > abs(result) || indexInput == -1)
				{
					result = input[newIndex];
					indexInput = newIndex;
				}
			}

			pooled[index] = result;
			pooling[indexInput] = 1.0;
		}
		else if (pt == Global::PoolingType::Average)
		{
			double ratio = 1.0 / (double)count;
			double result = 0;

			for (int i = 0; i < count; i++)
			{
				int newIndex = indexBatch * sizeFilter * sizeRow * sizeUnpooled +
					indexFilter * sizeRow * sizeUnpooled +
					indexRow * sizeUnpooled +
					indexColumn * sizePooling + i;
				pooling[newIndex] = ratio;

				result += input[newIndex];
			}

			pooled[index] = result / (double)count;
		}
		else
		{
		}
	}
}

void Calculation::Calculation_Pooling_1D_Forward(
	int sizeBatch, int sizeRow, int sizeFilter, int sizeUnpooled,
	double * input, int sizePooling, int sizePooled, Global::PoolingType pt,
	double* pooled, double* pooling)
{
	cuda_Calculation_Pooling_1D_Forward
		<< <Cuda::CUDA_Get_Block_Size(sizeBatch * sizeRow * sizeFilter * sizePooled), CUDA_CALCULATION_BLOCK_THREAD_SIZE >> >
		(sizeBatch, sizeRow, sizeFilter, sizeUnpooled, input,
			sizePooling, sizePooled, pt, pooled, pooling);
	Cuda::CUDA_Calculation_Error();
}

__global__ void cuda_Calculation_Pooling_1D_Backward(
	int sizeBatch, int sizeRow, int sizeFilter, int sizeUnpooled,
	double * unpooled_signal, int sizePooling, int sizePooled,
	double * pooled_signal, double* pooling)
{
	int index = blockDim.x * blockIdx.x + threadIdx.x;

	int indexBatch = index;

	int indexColumn = indexBatch % sizeUnpooled;
	indexBatch = indexBatch / sizeUnpooled;

	int indexRow = indexBatch % sizeRow;
	indexBatch = indexBatch / sizeRow;

	int indexFilter = indexBatch % sizeFilter;
	indexBatch = indexBatch / sizeFilter;

	if (indexBatch < sizeBatch)
	{
		unpooled_signal[index] = pooling[index] *
			pooled_signal[
				indexBatch * sizeFilter * sizeRow * sizePooled +
					indexFilter * sizeRow * sizePooled +
					indexRow * sizePooled +
					indexColumn / sizePooling];
	}
}

void Calculation::Calculation_Pooling_1D_Backward(
	int sizeBatch, int sizeRow, int sizeFilter, int sizeUnpooled,
	double * unpooled_signal, int sizePooling, int sizePooled,
	double * pooled_signal, double* pooling)
{
	cuda_Calculation_Pooling_1D_Backward
		<< <Cuda::CUDA_Get_Block_Size(sizeBatch * sizeRow * sizeFilter * sizeUnpooled), CUDA_CALCULATION_BLOCK_THREAD_SIZE >> >
		(sizeBatch, sizeRow, sizeFilter, sizeUnpooled, unpooled_signal,
			sizePooling, sizePooled, pooled_signal, pooling);
	Cuda::CUDA_Calculation_Error();
}


// Convolution 2D calculation layer

__global__ void cuda_Calculation_Convolution_2D_Forward(
	int sizeBatch, int inputFilter, int inputHeight, int inputWidth,
	double* input, int windowHeight, int windowWidth, double* weight, double* bias,
	Global::ActivationFunction af, double* activated,
	int outputFilter, int outputHeight, int outputWidth)
{
	int index = blockDim.x * blockIdx.x + threadIdx.x;

	int indexBatch = index;

	int indexOutputWidth = indexBatch % outputWidth;
	indexBatch = indexBatch / outputWidth;

	int indexOutputHeight = indexBatch % outputHeight;
	indexBatch = indexBatch / outputHeight;

	int indexOutputFilter = indexBatch % outputFilter;
	indexBatch = indexBatch / outputFilter;

	if (indexBatch < sizeBatch)
	{
		double result = 0.0;

		for (int i = 0; i < inputFilter; i++)
		{
			for (int j = 0; j < windowHeight; j++)
			{
				for (int k = 0; k < windowWidth; k++)
				{
					result +=
						input[indexBatch * inputFilter * inputHeight * inputWidth +
								i * inputHeight * inputWidth +
								(indexOutputHeight + j) * inputWidth +
								indexOutputWidth + k] *
						weight[indexOutputFilter * inputFilter * windowHeight * windowWidth +
								i * windowHeight * windowWidth +
								j * windowWidth +
								k];
				}
			}
		}

		if (bias == nullptr)
		{
			activated[index] = cuda_activation(result, af);
		}
		else
		{
			activated[index] = cuda_activation(result + bias[indexOutputFilter], af);
		}
	}
}

void Calculation::Calculation_Convolution_2D_Forward(
	int sizeBatch, int inputFilter, int inputHeight, int inputWidth,
	double * input, int windowHeight, int windowWidth, double * weight, double * bias,
	Global::ActivationFunction af, int outputFilter, int outputHeight, int outputWidth,
	double * output)
{
	cuda_Calculation_Convolution_2D_Forward
		<< <Cuda::CUDA_Get_Block_Size(sizeBatch * outputFilter * outputHeight * outputWidth), CUDA_CALCULATION_BLOCK_THREAD_SIZE >> >
		(sizeBatch, inputFilter, inputHeight, inputWidth, input,
			windowHeight, windowWidth, weight, bias, af, output,
			outputFilter, outputHeight, outputWidth);
	Cuda::CUDA_Calculation_Error();
}

__global__ void cuda_Calculation_Convolution_2D_Backward_Derivative_Bias(
	int sizeBatch, int sizeFilter, int sizeHeight, int sizeWidth,
	double * bias_grad, Global::ActivationFunction af,
	double * activiated, double * o_signal, int threadPerResult)
{
	extern __shared__ double buffer[];
	int index = blockDim.x * blockIdx.x + threadIdx.x;

	int indexFilter = 0;
	int indexCalculation = 0;

	if (!cuda_Reduce_Setup(index, sizeFilter, threadPerResult,
		&indexFilter, &indexCalculation))
	{
		if (bias_grad != nullptr)
		{
			cuda_Reduce_Dummy(threadPerResult);
		}
		return;
	}

	double result = 0.0;

	int sizeCalculation = sizeBatch * sizeHeight * sizeWidth;

	for (int i = indexCalculation; i < sizeCalculation; i += threadPerResult)
	{
		int indexBatch = i;

		int indexWidth = indexBatch % sizeWidth;
		indexBatch /= sizeWidth;

		int indexHeight = indexBatch % sizeHeight;
		indexBatch /= sizeHeight;

		int indexResult =
			indexBatch * sizeFilter * sizeHeight * sizeWidth +
			indexFilter * sizeHeight * sizeWidth +
			indexHeight * sizeWidth +
			indexWidth;

		double tempSignal = o_signal[indexResult] * cuda_derivative(activiated[indexResult], af);

		if (bias_grad != nullptr)
		{
			result += tempSignal;
		}

		o_signal[indexResult] = tempSignal;
	}

	if (bias_grad != nullptr)
	{
		cuda_Reduce(&buffer[threadIdx.x - threadIdx.x % threadPerResult], result, threadIdx.x, threadPerResult, &bias_grad[indexFilter]);
	}
}

__global__ void cuda_Calculation_Convolution_2D_Backward_Signal(
	int sizeBatch, int inputFilter, int inputHeight, int inputWidth,
	int windowHeight, int windowWidth, int outputFilter, int outputHeight, int outputWidth,
	double* i_signal, double* weight, double* o_signal)
{
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	int indexBatch = index;

	int indexInputWidth = indexBatch % inputWidth;
	indexBatch /= inputWidth;

	int indexInputHeight = indexBatch % inputHeight;
	indexBatch /= inputHeight;

	int indexInputFilter = indexBatch % inputFilter;
	indexBatch /= inputFilter;

	if (indexBatch < sizeBatch)
	{
		double result = 0.0;

		for (int indexOutputFilter = 0; indexOutputFilter < outputFilter; indexOutputFilter++)
		{
			for (int i = 0; i < windowHeight; i++)
			{
				for (int j = 0; j < windowWidth; j++)
				{
					int indexOutputHeight = indexInputHeight - i;
					int indexOutputWidth = indexInputWidth - j;
					if (indexOutputHeight >= 0 && indexOutputHeight < outputHeight && indexOutputWidth >= 0 && indexOutputWidth < outputWidth)
					{
						result +=
							o_signal[
								indexBatch * outputFilter * outputHeight * outputWidth +
									indexOutputFilter * outputHeight * outputWidth +
									indexOutputHeight * outputWidth +
									indexOutputWidth] *
							weight[
								indexOutputFilter * inputFilter * windowHeight * windowWidth +
									indexInputFilter * windowHeight * windowWidth +
									i * windowWidth +
									j];
					}
				}
			}
		}

		i_signal[index] = result;
	}
}

__global__ void cuda_Calculation_Convolution_2D_Backward_Weight_Grad(
	int sizeBatch, int inputFilter, int inputHeight, int inputWidth,
	int windowHeight, int windowWidth, int outputFilter, int outputHeight, int outputWidth,
	double * input, double* weight_grad, double* o_signal, int threadPerResult)
{
	extern __shared__ double buffer[];
	int index = blockDim.x * blockIdx.x + threadIdx.x;

	int indexWeight = 0;
	int indexCalculation = 0;
	if (!cuda_Reduce_Setup(index, outputFilter * inputFilter * windowHeight * windowWidth, threadPerResult, &indexWeight, &indexCalculation))
	{
		cuda_Reduce_Dummy(threadPerResult);
		return;
	}

	int indexOutputFilter = indexWeight;

	int indexWindowWidth = indexOutputFilter % windowWidth;
	indexOutputFilter /= windowWidth;

	int indexWindowHeight = indexOutputFilter % windowHeight;
	indexOutputFilter /= windowHeight;

	int indexInputFilter = indexOutputFilter % inputFilter;
	indexOutputFilter /= inputFilter;

	double result = 0.0;

	int sizeCalc = sizeBatch * outputHeight * outputWidth;
	for (int indexCalc = indexCalculation; indexCalc < sizeCalc; indexCalc += threadPerResult)
	{
		int indexBatch = indexCalc;

		int indexOutputWidth = indexCalc % outputWidth;
		indexBatch /= outputWidth;

		int indexOutputHeight = indexCalc % outputHeight;
		indexBatch /= outputHeight;

		result +=
			o_signal[
				indexBatch * outputFilter * outputHeight * outputWidth +
					indexOutputFilter * outputHeight * outputWidth +
					indexOutputHeight * outputWidth +
					indexOutputWidth] *
			input[
				indexBatch * inputFilter * inputHeight * inputWidth +
					indexInputFilter * inputHeight * inputWidth +
					(indexOutputHeight + indexWindowHeight) * inputWidth +
					indexOutputWidth + indexWindowWidth];
	}

	cuda_Reduce(&buffer[threadIdx.x - threadIdx.x % threadPerResult], result, threadIdx.x, threadPerResult, &weight_grad[indexWeight]);
}

void Calculation::Calculation_Convolution_2D_Backward(
	int sizeBatch, int inputFilter, int inputHeight, int inputWidth,
	double* i_signal, double* input, int windowHeight, int windowWidth,
	double** weight, double** bias, Global::ActivationFunction af,
	int outputFilter, int outputHeight, int outputWidth, double* activated, double* o_signal,
	CudaParameter parameterBias, CudaParameter parameterWeight)
{
	cuda_Calculation_Convolution_2D_Backward_Derivative_Bias
		<< <parameterBias.CountBlock, CUDA_CALCULATION_BLOCK_THREAD_SIZE, CUDA_CALCULATION_BLOCK_THREAD_SIZE * sizeof(double) >> >
		(sizeBatch, outputFilter, outputHeight, outputWidth,
			bias[0] == nullptr ? nullptr : _processingData1, af,
			activated, o_signal, parameterBias.ThreadPerResult);
	Cuda::CUDA_Calculation_Error();

	if (bias[0] != nullptr)
	{
		update_by_grad(
			outputFilter, _processingData1, bias,
			nullptr, 0, _learning_rate, _update_method);
	}

	if (i_signal)
	{
		cuda_Calculation_Convolution_2D_Backward_Signal
			<< <Cuda::CUDA_Get_Block_Size(sizeBatch * inputFilter * inputHeight * inputWidth), CUDA_CALCULATION_BLOCK_THREAD_SIZE >> >
			(sizeBatch, inputFilter, inputHeight, inputWidth, windowHeight, windowWidth,
				outputFilter, outputHeight, outputWidth, i_signal, weight[0], o_signal);
		Cuda::CUDA_Calculation_Error();
	}

	cuda_Calculation_Convolution_2D_Backward_Weight_Grad
		<< <parameterWeight.CountBlock, CUDA_CALCULATION_BLOCK_THREAD_SIZE, CUDA_CALCULATION_BLOCK_THREAD_SIZE * sizeof(double) >> >
		(sizeBatch, inputFilter, inputHeight, inputWidth, windowHeight, windowWidth,
			outputFilter, outputHeight, outputWidth, input, _processingData1, o_signal,
			parameterWeight.ThreadPerResult);
	Cuda::CUDA_Calculation_Error();

	update_by_grad(
		outputFilter * inputFilter * windowHeight * windowWidth, _processingData1,
		weight, nullptr, 0, _learning_rate, _update_method);
}


// Convolutional 2D pooling layer

__global__ void cuda_Calculation_Pooling_2D_Forward(
	int sizeBatch, int sizeFilter, int unpooledHeight, int unpooledWidth,
	double* input, int poolingHeight, int poolingWidth, int pooledHeight, int pooledWidth, 
	Global::PoolingType pt,	double* pooled, double* pooling)
{
	int index = blockDim.x * blockIdx.x + threadIdx.x;

	int indexBatch = index;

	int indexWidth = indexBatch % pooledWidth;
	indexBatch /= pooledWidth;

	int indexHeight = indexBatch % pooledHeight;
	indexBatch /= pooledHeight;

	int indexFilter = indexBatch % sizeFilter;
	indexBatch /= sizeFilter;

	if (indexBatch < sizeBatch)
	{
		int countHeigth = min(poolingHeight, unpooledHeight - indexHeight * poolingHeight);
		int countWidth = min(poolingWidth, unpooledWidth - indexWidth * poolingWidth);

		if (pt == Global::PoolingType::AbsoluteMax)
		{
			int indexInput = -1;
			double result = 0;

			for (int i = 0; i < countHeigth; i++)
			{
				for (int j = 0; j < countWidth; j++)
				{
					int newIndex = indexBatch * sizeFilter * unpooledHeight * unpooledWidth +
						indexFilter * unpooledHeight * unpooledWidth +
						(indexHeight * poolingHeight + i) * unpooledWidth +
						indexWidth * poolingWidth + j;
					pooling[newIndex] = 0.0;

					if (abs(input[newIndex]) > abs(result) || indexInput == -1)
					{
						result = input[newIndex];
						indexInput = newIndex;
					}
				}
			}

			pooled[index] = result;
			pooling[indexInput] = 1.0;
		}
		else if (pt == Global::PoolingType::Average)
		{
			double ratio = 1.0 / (double)(countHeigth * countWidth);
			double result = 0;

			for (int i = 0; i < countHeigth; i++)
			{
				for (int j = 0; j < countWidth; j++)
				{
					int newIndex = indexBatch * sizeFilter * unpooledHeight * unpooledWidth +
						indexFilter * unpooledHeight * unpooledWidth +
						(indexHeight * poolingHeight + i) * unpooledWidth +
						indexWidth * poolingWidth + j;
					pooling[newIndex] = ratio;

					result += input[newIndex];
				}
			}

			pooled[index] = result / (double)(countHeigth * countWidth);
		}
		else
		{
		}
	}
}

void Calculation::Calculation_Pooling_2D_Forward(
	int sizeBatch, int sizeFilter, int unpooledHeight, int unpooledWidth,
	double * input, int poolingHeight, int poolingWidth,
	int pooledHeight, int pooledWidth, Global::PoolingType pt, double* pooled, double* pooling)
{
	cuda_Calculation_Pooling_2D_Forward
		<< <Cuda::CUDA_Get_Block_Size(sizeBatch * sizeFilter * pooledHeight * pooledWidth), CUDA_CALCULATION_BLOCK_THREAD_SIZE >> >
		(sizeBatch, sizeFilter, unpooledHeight, unpooledWidth, input,
			poolingHeight, poolingWidth, pooledHeight, pooledWidth, pt, pooled, pooling);
	Cuda::CUDA_Calculation_Error();
}

__global__ void cuda_Calculation_Pooling_2D_Backward(
	int sizeBatch, int sizeFilter, int unpooledHeight, int unpooledWidth,
	double * unpooled_signal, int poolingHeight, int poolingWidth, int pooledHeight, int pooledWidth,
	double * pooled_signal, double* pooling)
{
	int index = blockDim.x * blockIdx.x + threadIdx.x;

	int indexBatch = index;

	int indexWidth = indexBatch % unpooledWidth;
	indexBatch = indexBatch / unpooledWidth;

	int indexHeight = indexBatch % unpooledHeight;
	indexBatch = indexBatch / unpooledHeight;

	int indexFilter = indexBatch % sizeFilter;
	indexBatch = indexBatch / sizeFilter;

	if (indexBatch < sizeBatch)
	{
		unpooled_signal[index] = pooling[index] *
			pooled_signal[
				indexBatch * sizeFilter * pooledHeight * pooledWidth +
					indexFilter * pooledHeight * pooledWidth +
					(indexHeight / poolingHeight) * pooledWidth +
					indexWidth / poolingWidth];
	}
}

void Calculation::Calculation_Pooling_2D_Backward(
	int sizeBatch, int sizeFilter, int unpooledHeight, int unpooledWidth,
	double * unpooled_signal, int poolingHeight, int poolingWidth,
	int pooledHeight, int pooledWidth, double * pooled_signal, double* pooling)
{
	cuda_Calculation_Pooling_2D_Backward
		<< <Cuda::CUDA_Get_Block_Size(sizeBatch * sizeFilter * unpooledHeight * unpooledWidth), CUDA_CALCULATION_BLOCK_THREAD_SIZE >> >
		(sizeBatch, sizeFilter, unpooledHeight, unpooledWidth, unpooled_signal,
			poolingHeight, poolingWidth, pooledHeight, pooledWidth, pooled_signal, pooling);
	Cuda::CUDA_Calculation_Error();
}


 // Fully-connected calculation layer

__global__ void cuda_Calculation_FullyConnected_Forward(
	int sizeBatch, int sizeInput, int sizeOutput, int* dropoutSelect,
	double* input, double* weight, double scaleRatio, double* bias,
	Global::ActivationFunction af, double* activated, int threadPerResult)
{
	extern __shared__ double buffer[];
	int index = blockDim.x * blockIdx.x + threadIdx.x;

	int indexOutputBatch = 0;
	int indexCalculation = 0;
	if (!cuda_Reduce_Setup(index, sizeBatch * sizeOutput, threadPerResult, &indexOutputBatch, &indexCalculation))
	{
		cuda_Reduce_Dummy(threadPerResult);
		return;
	}

	int indexBatch = indexOutputBatch / sizeOutput;
	int indexOutput = indexOutputBatch % sizeOutput;
	double result = 0.0;

	for (int i = indexCalculation; i < sizeInput; i += threadPerResult)
	{
		if (dropoutSelect == nullptr ? true : dropoutSelect[i])
		{
			result += input[indexBatch * sizeInput + i] * weight[indexOutput * sizeInput + i];
		}
	}

	cuda_Reduce(buffer, result, threadIdx.x, threadPerResult, &result);

	if (indexCalculation == 0)
	{
		if (bias == nullptr)
		{
			activated[indexBatch * sizeOutput + indexOutput] =
				cuda_activation(result * scaleRatio, af);
		}
		else
		{
			activated[indexBatch * sizeOutput + indexOutput] =
				cuda_activation(result * scaleRatio + bias[indexOutput], af);
		}
	}
}

void Calculation::Calculation_FullyConnected_Forward(
	int sizeBatch, int sizeInput, double * input,
	int * dropoutSelectInput, double scale, double ** weight, double ** bias, 
	Global::ActivationFunction af, int sizeOutput, double * activated, 
	CudaParameter parameter)
{
	cuda_Calculation_FullyConnected_Forward
		<< <parameter.CountBlock, CUDA_CALCULATION_BLOCK_THREAD_SIZE, CUDA_CALCULATION_BLOCK_THREAD_SIZE * sizeof(double) >> >
		(sizeBatch, sizeInput, sizeOutput, dropoutSelectInput, input, 
			weight[0], scale, bias[0], af, activated, parameter.ThreadPerResult);
	Cuda::CUDA_Calculation_Error();
}

__global__ void cuda_Calculation_FullyConnected_Backward_Derivative_Bias(
	int sizeBatch, int sizeOutput, double * bias_grad, 
	Global::ActivationFunction af, double * activiated, double * o_signal,
	int threadPerResult)
{           
	extern __shared__ double buffer[];
	int index = blockDim.x * blockIdx.x + threadIdx.x;

	int indexGroup = 0;
	int indexCalculation = 0;

	if (!cuda_Reduce_Setup(index, sizeOutput, threadPerResult, &indexGroup, &indexCalculation))
	{
		if (bias_grad != nullptr)
		{
			cuda_Reduce_Dummy(threadPerResult);
		}
		return;
	}

	double result = 0.0;

	for (int i = indexCalculation; i < sizeBatch; i += threadPerResult)
	{
		int indexResult = i * sizeOutput + indexGroup;

		double tempSignal = o_signal[indexResult] * cuda_derivative(activiated[indexResult], af);

		if (bias_grad != nullptr)
		{
			result += tempSignal;
		}
		o_signal[indexResult] = tempSignal;
	}

	if (bias_grad != nullptr)
	{
		cuda_Reduce(&buffer[threadIdx.x - threadIdx.x % threadPerResult], result, threadIdx.x, threadPerResult, &bias_grad[indexGroup]);
	}
}

__global__ void cuda_Calculation_FullyConnected_Backward_Signal(
	int sizeBatch, int sizeInput, int * dropoutSelectOutput, double* i_signal, 
	double* weight,	int sizeOutput, double* o_signal, int threadPerResult)
{
	extern __shared__ double buffer[];
	int index = blockDim.x * blockIdx.x + threadIdx.x;

	int indexGroup = 0;
	int indexOutput = 0;

	if (!cuda_Reduce_Setup(index, sizeBatch * sizeInput, threadPerResult,
		&indexGroup, &indexOutput))
	{
		cuda_Reduce_Dummy(threadPerResult);
		return;
	}

	int indexBatch = indexGroup / sizeInput;
	int indexInput = indexGroup % sizeInput;

	double result = 0.0;

	for (int i = indexOutput; i < sizeOutput; i += threadPerResult)
	{
		if (dropoutSelectOutput == nullptr || dropoutSelectOutput[i])
		{
			result += o_signal[indexBatch * sizeOutput + i] * weight[i * sizeInput + indexInput];
		}
	}
	
	cuda_Reduce(&buffer[threadIdx.x - threadIdx.x % threadPerResult], result, threadIdx.x, threadPerResult, &i_signal[indexGroup]);
}

__global__ void cuda_Calculation_FullyConnected_Backward_Weight_Grad(
	int sizeBatch, int sizeInput, double * input, double* weight_grad, 
	int sizeOutput, double* o_signal, int threadPerResult)
{
	extern __shared__ double buffer[];
	int index = blockDim.x * blockIdx.x + threadIdx.x;

	int indexWeight = 0;
	int indexBatch = 0;
	if (!cuda_Reduce_Setup(index, sizeInput * sizeOutput, threadPerResult, &indexWeight, &indexBatch))
	{
		cuda_Reduce_Dummy(threadPerResult);
		return;
	}

	int indexOutput = indexWeight / sizeInput;
	int indexInput = indexWeight % sizeInput;

	double result = 0.0;

	for (int i = indexBatch; i < sizeBatch; i += threadPerResult)
	{
		result += o_signal[i * sizeOutput + indexOutput] * input[i * sizeInput + indexInput];
	}

	cuda_Reduce(&buffer[threadIdx.x - threadIdx.x % threadPerResult], result, threadIdx.x, threadPerResult, &weight_grad[indexWeight]);
}

void Calculation::Calculation_FullyConnected_Backward(
	int sizeBatch, int sizeInput, double* i_signal, double * input, 
	double ** weights, double ** bias, Global::ActivationFunction af, 
	int sizeOutput, double* activated, int * dropoutSelectOutput,
	double* o_signal, CudaParameter parameterBias, 
	CudaParameter parameterSignal, CudaParameter parameterWeight)
{

	cuda_Calculation_FullyConnected_Backward_Derivative_Bias
		<< <parameterBias.CountBlock, CUDA_CALCULATION_BLOCK_THREAD_SIZE, CUDA_CALCULATION_BLOCK_THREAD_SIZE * sizeof(double) >> >
		(sizeBatch, sizeOutput, 
			bias[0] == nullptr ? nullptr : _processingData1, af,
			activated, o_signal, parameterBias.ThreadPerResult);
	Cuda::CUDA_Calculation_Error();

	if (bias[0] != nullptr)
	{
		update_by_grad(
			sizeOutput, _processingData1, bias,
			dropoutSelectOutput, sizeOutput, _learning_rate, _update_method);
	}

	if (i_signal)
	{
		cuda_Calculation_FullyConnected_Backward_Signal
			<< <parameterSignal.CountBlock, CUDA_CALCULATION_BLOCK_THREAD_SIZE, CUDA_CALCULATION_BLOCK_THREAD_SIZE * sizeof(double) >> >
			(sizeBatch, sizeInput, dropoutSelectOutput, i_signal, weights[0], sizeOutput, o_signal, parameterSignal.ThreadPerResult);
		Cuda::CUDA_Calculation_Error();
	}

	cuda_Calculation_FullyConnected_Backward_Weight_Grad
		<< <parameterWeight.CountBlock, CUDA_CALCULATION_BLOCK_THREAD_SIZE, CUDA_CALCULATION_BLOCK_THREAD_SIZE * sizeof(double) >> >
		(sizeBatch, sizeInput, input, _processingData1, sizeOutput, o_signal, parameterWeight.ThreadPerResult);
	Cuda::CUDA_Calculation_Error();

	update_by_grad(
		sizeOutput * sizeInput, _processingData1, weights,
		dropoutSelectOutput, sizeOutput, _learning_rate, _update_method);

}



// Normalization update functions

__global__ void cuda_Normalize_Update(
	double* running, double* current, double* root, int count, double runningScale)
{
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	if (index >= count) { return; }

	if (!running)
	{
		root[index] = sqrt(current[index] + CUDA_CALCULATION_NEAR_ZERO);
	}

	double new_var = (running[index] *
		(1.0 - runningScale / CUDA_CALCULATION_NORMALIZING_RATIO) *
		CUDA_CALCULATION_NORMALIZING_RATIO +
		current[index] * (1.0 - CUDA_CALCULATION_NORMALIZING_RATIO)) /
		(1.0 - runningScale);
	running[index] = new_var;

	if (root)
	{
		root[index] = sqrt(new_var + CUDA_CALCULATION_NEAR_ZERO);
	}
}


// Convolution layer forward

__global__ void cuda_Normalize_Convolution_Mean(
	double* data, int sizeBatch, int countFilter, int countRow, int sizeRow,
	double* mean, int threadPerResult)
{
	extern __shared__ double buffer[];
	int index = blockDim.x * blockIdx.x + threadIdx.x;

	int indexGroup = 0;
	int indexCalculation = 0;
	if (!cuda_Reduce_Setup(index, countFilter * countRow, threadPerResult,
		&indexGroup, &indexCalculation))
	{
		cuda_Reduce_Dummy(threadPerResult);
		return;
	}

	int indexRow = indexGroup % countRow;
	int indexFilter = indexGroup / countRow;

	// Add up data
	double result = 0.0;
	int count = sizeBatch * sizeRow;
	for (int i = indexCalculation; i < count; i += threadPerResult)
	{
		int indexBatch = i / sizeRow;
		int indexData = i % sizeRow;
		result += data[
			indexBatch * countFilter * countRow * sizeRow + 
				indexFilter * countRow * sizeRow +
				indexRow * sizeRow + indexData] / (double)(count);
	}

	cuda_Reduce(&buffer[threadIdx.x - threadIdx.x % threadPerResult], result, threadIdx.x, threadPerResult, &mean[indexGroup]);
}

__global__ void cuda_Normalize_Convolution_Variance(
	double* data, int sizeBatch, int countFilter, int countRow, int sizeRow,
	double* mean, double* variance, int threadPerResult)
{
	extern __shared__ double buffer[];
	int index = blockDim.x * blockIdx.x + threadIdx.x;

	int indexGroup = 0;
	int indexCalculation = 0;
	if (!cuda_Reduce_Setup(index, countFilter * countRow, threadPerResult,
		&indexGroup, &indexCalculation))
	{
		cuda_Reduce_Dummy(threadPerResult);
		return;
	}

	int indexRow = indexGroup % countRow;
	int indexFilter = indexGroup / countRow;

	// Add up data
	double result = 0.0;
	int count = sizeBatch * sizeRow;
	for (int i = indexCalculation; i < count; i += threadPerResult)
	{
		int indexBatch = i / sizeRow;
		int indexData = i % sizeRow;
		result += (data[
			indexBatch * countFilter * countRow * sizeRow +
				indexFilter * countRow * sizeRow +
				indexRow * sizeRow + indexData] - mean[indexGroup]) / (double)(count);
	}

	cuda_Reduce(&buffer[threadIdx.x - threadIdx.x % threadPerResult], result, threadIdx.x, threadPerResult, &variance[indexGroup]);
}

__global__ void cuda_Convolution_Normalize(
	int sizeBatch, int countFilter, int countRow, int sizeRow,
	double* data, double* mean, double* sd, double* normalized,
	int threadPerResult)
{
	int index = blockDim.x * blockIdx.x + threadIdx.x;

	int indexGroup = 0;
	int indexCalculation = 0;
	if (!cuda_Reduce_Setup(index, countFilter * countRow, threadPerResult,
		&indexGroup, &indexCalculation))
	{
		return;
	}

	int indexRow = indexGroup % countRow;
	int indexFilter = indexGroup / countRow;

	for (int i = indexCalculation; i < sizeBatch * sizeRow; i += threadPerResult)
	{
		int indexBatch = i / sizeRow;
		int indexData = i % sizeRow;
		int indexMod =
			((indexBatch * countFilter + indexFilter) *
				countRow + indexRow) * sizeRow + indexData;

		normalized[indexMod] = (data[indexMod] - mean[indexGroup]) / sd[indexGroup];
	}
}

void Calculation::Convolution_Normalize_Convolution(
	int sizeBatch, int countFilter, int countRow, int sizeRow, double * data,
	double * mean, double * variance, double * sd, double * normalized, 
	CudaParameter parameter)
{
	cuda_Normalize_Convolution_Mean
		<< < parameter.CountBlock, CUDA_CALCULATION_BLOCK_THREAD_SIZE, CUDA_CALCULATION_BLOCK_THREAD_SIZE * sizeof(double) >> >
		(data, sizeBatch, countFilter, countRow, sizeRow,
			_processingData1, parameter.ThreadPerResult);
	Cuda::CUDA_Calculation_Error();

	cuda_Normalize_Update
		<< < countFilter * countRow, CUDA_CALCULATION_BLOCK_THREAD_SIZE >> >
		(mean, _processingData1, nullptr, countFilter * countRow, _normalizing_running_ratio);
	Cuda::CUDA_Calculation_Error();

	cuda_Normalize_Convolution_Variance
		<< < parameter.CountBlock, CUDA_CALCULATION_BLOCK_THREAD_SIZE, CUDA_CALCULATION_BLOCK_THREAD_SIZE * sizeof(double) >> >
		(data, sizeBatch, countFilter, countRow, sizeRow,
			mean, variance, parameter.ThreadPerResult);
	Cuda::CUDA_Calculation_Error();

	cuda_Normalize_Update
		<< < countFilter * countRow, CUDA_CALCULATION_BLOCK_THREAD_SIZE >> >
		(mean, _processingData1, sd, countFilter * countRow, _normalizing_running_ratio);
	Cuda::CUDA_Calculation_Error();

	cuda_Convolution_Normalize
		<< < parameter.CountBlock, CUDA_CALCULATION_BLOCK_THREAD_SIZE >> >
		(sizeBatch, countFilter, countRow, sizeRow, data, mean, sd, normalized, parameter.ThreadPerResult);
	Cuda::CUDA_Calculation_Error();
}

