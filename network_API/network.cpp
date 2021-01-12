#include "network.h"
#include "cuda.cuh"
#include "calculation.cuh"
#include <stdio.h>
#include <chrono>

Network::Network(char* iData, int iLength, int iRowSize, int iRow, int iOutputSizeVerify, int iMaxBatch, double iLearningRate,
	Global::ErrorFunction iErrorFunction, Global::UpdateMethod iUpdateMethod, void(*iLogMessage)(char*))
{
	char message[256];

	_layerFirst = new Input(iRow, iRowSize, iMaxBatch);
	_layerLast = _layerFirst;

	_sizeInput = iRow * iRowSize;
	_sizeOutput = iOutputSizeVerify;
	_maxBatch = iMaxBatch;

	_trainingSet = 0;
	_trainingCount = 0;

	_errorFunction = iErrorFunction;

	_expected = Cuda::CUDA_Array_Allocate<double>(_sizeOutput * _maxBatch);

	int tempSize = CUDA_CALCULATION_MINIMAL_THREADS;

	sprintf_s(message, 256, "*** Creating network ***\n\n");
	iLogMessage(message);

	for (int i = 0; i < iLength; i++)
	{
		int index = i * 40;

		Global::LayerType type = (Global::LayerType)*((int*)(&iData[index]));
		bool bool01 = *((bool*)(&iData[index + 4]));
		int int01 = *((int*)(&iData[index + 8]));
		int int02 = *((int*)(&iData[index + 12]));
		int int03 = *((int*)(&iData[index + 16]));
		int int04 = *((int*)(&iData[index + 20]));
		double double01 = *((double*)(&iData[index + 24]));

		int lastSize = _layerLast->Size();

		switch (type)
		{
		case Global::LayerType::CalculationC_1D:
			_layerLast = new CalculationC_1D(_layerLast, int01, bool01, (Global::ActivationFunction)int02, int03);
			sprintf_s(
				message, 256, 
				"1D Convoluation Layer: { windowSize: %d, bias: %d, activation: %d, filter: %d } -> %d x %d x %d -> %d\n", 
				int01, bool01, int02, int03, 
				_layerLast->ConvRowSize(), _layerLast->ConvRowCount(), _layerLast->ConvFilter(), _layerLast->Size());
			iLogMessage(message);
			break;
		case Global::LayerType::Pooling_1D:
			_layerLast = new Pooling_1D(_layerLast, int01, (Global::PoolingType)int02);
			sprintf_s(
				message, 256, 
				"1D Pooling Layer: { size: %d, pooling: %d } -> %d x %d x %d -> %d\n", 
				int01, int02,
				_layerLast->ConvRowSize(), _layerLast->ConvRowCount(), _layerLast->ConvFilter(), _layerLast->Size());
			iLogMessage(message);
			break;
		case Global::LayerType::CalculationC_2D:
			_layerLast = new CalculationC_2D(_layerLast, int01, int02, bool01, (Global::ActivationFunction)int03, int04);
			sprintf_s(
				message, 256, 
				"2D Convoluation Layer: { windowSize: %d/%d, bias: %d, activation: %d, filter: %d } -> %d x %d x %d -> %d\n",
				int01, int02, bool01, int03, int04,
				_layerLast->ConvRowSize(), _layerLast->ConvRowCount(), _layerLast->ConvFilter(), _layerLast->Size());
			iLogMessage(message);
			break;
		case Global::LayerType::Pooling_2D:
			_layerLast = new Pooling_2D(_layerLast, int01, int02, (Global::PoolingType)int03);
			sprintf_s(
				message, 256, 
				"2D Pooling Layer: { size: %d/%d, pooling: %d } -> %d x %d x %d -> %d\n", 
				int01, int02, int03,
				_layerLast->ConvRowSize(), _layerLast->ConvRowCount(), _layerLast->ConvFilter(), _layerLast->Size());
			iLogMessage(message);
			break;
		case Global::LayerType::CalculationFC:
			_layerLast = new CalculationFC(_layerLast, bool01, double01, (Global::ActivationFunction)int01, int02);
			sprintf_s(
				message, 256, 
				"Fully-Conntected Layer: { bias: %d, dropout: %f, activation: %d, outputSize: %d } -> %d\n", 
				bool01, double01, int01, int02, _layerLast->Size());
			iLogMessage(message);
			tempSize = max(tempSize, _layerLast->Size() * lastSize);
			break;
		default:
			sprintf_s(message, 256, "");
			break;
		}

		tempSize = max(tempSize, _layerLast->Size() * _maxBatch);
	}

	if (_layerLast->Size() != iOutputSizeVerify)
	{
		throw "The output size of constructed network mismatches with the desired output.";
	}

	sprintf_s(message, 256, "\nProcessing CUDA array size: %d\n\n", tempSize);
	iLogMessage(message);
	Calculation::Initialize(iLearningRate, 20, iUpdateMethod, tempSize);
}

Network::~Network()
{
	delete _layerLast;
	Cuda::CUDA_Array_Free((void**)&_expected);
}

static double * temp;

static void CUDA_Read_Data(double* cudaValues, double** target, int size, int offset)
{
	if (*target != nullptr) { delete *target; }
	*target = new double[size];
	Cuda::CUDA_Array_CopyDeviceToHost(&cudaValues[offset], size * sizeof(double), *target);
}

void Network::AddTrainingSet(double * iInput, double * iOutputExpected)
{
	Cuda::CUDA_Array_CopyDeviceToDevice(iInput, sizeof(double) * _sizeInput, &(_layerFirst->Output()[_sizeInput * _trainingSet]));
	Cuda::CUDA_Array_CopyDeviceToDevice(iOutputExpected, sizeof(double) * _sizeOutput, &(_expected[_sizeOutput * _trainingSet]));

	_trainingSet++;

	if (_trainingSet < _maxBatch)
	{
		return;
	}

	_layerLast->Forward(_trainingSet, true);

	Calculation::Backward_ErrorSignal(_layerLast->Output(), _expected, _layerLast->Signal(), _maxBatch * _sizeOutput, _errorFunction);
	_layerLast->Backward(_trainingSet, nullptr);

	_trainingSet = 0;
	_trainingCount++;
}

double Network::Error(double * iInput, double * iOutputExpected)
{
	Cuda::CUDA_Array_CopyDeviceToDevice(iInput, sizeof(double) * _sizeInput * _maxBatch, _layerFirst->Output());
	Cuda::CUDA_Array_CopyDeviceToDevice(iOutputExpected, sizeof(double) * _sizeOutput * _maxBatch, _expected);

	_layerLast->Forward(_maxBatch, false);
	double result = Calculation::Forward_Error(_layerLast->Output(), _expected, _maxBatch * _sizeOutput, _errorFunction);
	return result;
}

int Network::BatchTrained()
{
	return _trainingCount;
}

void Network::BatchTrainedReset()
{
	_trainingCount = 0;
}

void Network::Calculate(double * iInput, int count, double * oOutput)
{
	for (int i = 0; i < count; i += _maxBatch)
	{
		int batch = min(_maxBatch, count - i);
		Cuda::CUDA_Array_CopyHostToDevice(&iInput[i * _sizeInput], _sizeInput * batch * sizeof(double), _layerFirst->Output());
		_layerLast->Forward(batch, false);
		Cuda::CUDA_Array_CopyDeviceToHost(_layerLast->Output(), _sizeOutput * batch * sizeof(double), &oOutput[i * _sizeOutput]);
	}
}
